# Acausal operations. Everything here looks *forward* in time, violating the
# causal invariant the rest of the package upholds, so it lives in its own
# submodule and is reached only through `using CausalFrames.Acausal` — never
# re-exported from the top level. The forward join mirrors `asofjoin`'s
# streaming machinery (a chunkmap over the left stream pulling right chunks on
# demand, type-unstable setup once per right chunk, the per-row merge behind a
# function barrier), with the match direction, the tie-break, and the context
# widening all inverted.
module Acausal

using DataFrames
using Tables
using ..CausalFrames: CausalPipeline, Context, chunkmap, tokeycolumns,
    chunktypes, promotetypes, normprefix, prefixed,
    storerowtype, storekeytype, rowat, keyat, matchcolumn

export futurejoin

"""
    futurejoin(right::CausalPipeline; key = nothing, tolerance = nothing,
               strict = false, leftprefix = nothing, rightprefix = nothing,
               righttime = nothing) -> (CausalPipeline -> CausalPipeline)
    futurejoin(left::CausalPipeline, right::CausalPipeline; key = nothing,
               ...) -> CausalPipeline

The forward-looking mirror of [`asofjoin`](@ref CausalFrames.asofjoin): a
transform joining each left row to the **earliest** right row whose time is
not before the left row's time (`strict = true`: strictly after). Every left
row is kept; the right table's value columns are appended with element type
`Union{Missing, T}` and are `missing` where no right row qualifies. Among
right rows sharing one time, the first in stream order wins. The right `time`
column is dropped unless `righttime` names an output column for the matched
row's time.

This operator is **acausal**: a row emitted at time `t` looks at right rows
with time `>= t`. That is why it is segregated in the `CausalFrames.Acausal`
submodule and must be opted into explicitly (`using CausalFrames.Acausal`);
everything the top-level module exports stays causal.

With `key` (a column name or collection of column names, present in both
tables) rows join per unique key value, exact-matched; the key columns appear
once in the output, taken from the left row, never prefixed. With `tolerance`
the match additionally requires `rtime - time <= tolerance`, and the right
pipeline runs over the widened context `[start, stop + tolerance)` so
lookahead past the window end is covered — this requires the time type to
support addition (numbers and `Dates` types do). Without `tolerance` the
right pipeline sees only `[start, stop)`, so left rows near `stop` may find no
later right row.

`leftprefix` / `rightprefix` rename that side's non-time, non-key columns to
`"{prefix}_{name}"`. Output column names must be unique after prefixing — in
particular a self join (`p |> futurejoin(p)`) needs a prefix. A right stream
producing no chunks passes left chunks through unchanged (no right columns, no
`righttime`), except for the `leftprefix` rename.

Because the match is the earliest qualifying right row, buffering is
unavoidable: right rows are held per key until a left row consumes or outruns
them, and confirming that a key has *no* future match drains the right stream.
Worst-case memory is therefore O(number of right rows), unlike `asofjoin`'s
O(number of keys) store — the price of looking forward.

The curried form composes with `|>`; the uncurried form applies directly, so
`futurejoin(left, right; ...)` is equivalent to `left |> futurejoin(right; ...)`.
"""
function futurejoin(right::CausalPipeline; key = nothing, tolerance = nothing,
    strict::Bool = false, leftprefix = nothing,
    rightprefix = nothing,
    righttime::Union{Nothing,Symbol} = nothing)
    keycols = tokeycolumns(key)
    allunique(keycols) ||
        throw(ArgumentError("futurejoin key columns must be unique"))
    :time in keycols && throw(ArgumentError(
        "time is the as-of dimension and may not be a futurejoin key"))
    righttime === :time && throw(
        ArgumentError(
            "futurejoin righttime may not be time; it would collide with the left time column",
        ),
    )
    righttime !== nothing && righttime in keycols &&
        throw(ArgumentError(
            "futurejoin righttime $righttime collides with a key column"))
    lp = normprefix(leftprefix)
    rp = normprefix(rightprefix)
    return function (left::CausalPipeline)
        return CausalPipeline() do ctx::Context
            cfg = FutureJoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                strict ? (>) : (>=), lp, rp, righttime)
            js = FutureJoinState(right.run(futurecontext(ctx, tolerance)))
            return chunkmap(c -> joinchunk!(js, cfg, c), left.run(ctx))
        end
    end
end
futurejoin(left::CausalPipeline, right::CausalPipeline; kwargs...) =
    futurejoin(right; kwargs...)(left)

# The mirror of asofjoin's rightcontext: forward tolerance widens the right
# window forward, so lookahead past the window end is covered — the one place
# times are added. The explicit guard rejects a negative tolerance, which the
# Context constructor (only start <= stop) would otherwise accept.
futurecontext(ctx::Context, ::Nothing) = ctx
function futurecontext(ctx::Context, tolerance)
    stop = ctx.stop + tolerance
    stop >= ctx.stop || throw(ArgumentError(
        "futurejoin tolerance must be non-negative, got $tolerance"))
    return Context(ctx.start, stop)
end

# strict and tolerance ride in type parameters (`after` is `>` or `>=`), so
# the kernel specializes and neither costs a per-row branch.
struct FutureJoinConfig{KN,Tol,A}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    tolerance::Tol
    after::A
    leftprefix::Union{Nothing,String}
    rightprefix::Union{Nothing,String}
    righttime::Union{Nothing,Symbol}
end

# A per-key FIFO of buffered future right rows: append on pull, logical
# pop-front by advancing `head` (the segtree.jl idiom), with an amortized
# compaction so a buffer that stays small cannot grow its backing vector
# without bound. `matches` stores copied row values, not buffer indices, so
# compaction never invalidates an emitted match.
mutable struct KeyBuffer{V}
    rows::Vector{V}
    head::Int
end
KeyBuffer{V}() where {V} = KeyBuffer{V}(V[], 1)

@inline bufempty(b::KeyBuffer) = b.head > length(b.rows)
@inline buffront(b::KeyBuffer) = @inbounds b.rows[b.head]
@inline bufpush!(b::KeyBuffer, v) = (push!(b.rows, v); nothing)
function bufpop!(b::KeyBuffer)
    b.head += 1
    if b.head > length(b.rows)
        empty!(b.rows)
        b.head = 1
    elseif 2 * (b.head - 1) > length(b.rows)
        b.rows = b.rows[b.head:end]
        b.head = 1
    end
    return nothing
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the futuresegment! function barrier.
mutable struct FutureJoinState
    rchunks::Any       # right chunk iterator
    rstate::Any        # its iteration state
    rstarted::Bool
    rdone::Bool        # right stream exhausted
    rnt::Any           # current right column table (nothing until first pull)
    rpos::Int          # index of the next undrained right row in rnt
    rvaluenames::Any   # Vector{Symbol}: right columns minus keys minus time
    rtypes::Union{Nothing,NamedTuple}  # promotion of right schemas seen
    store::Any         # Dict{K,KeyBuffer{V}}: per-key buffered future rows
    matches::Any       # Vector{Union{Missing,V}}: per-left-row match, reused
    passthrough::Bool  # the right stream produced no chunks at all
    leftchecked::Bool  # key validation against the left schema done
    checked::Bool      # output-name duplicate validation done
    FutureJoinState(rchunks) = new(rchunks, nothing, false, false, nothing, 1,
        nothing, nothing, nothing, nothing, false,
        false, false)
end

function checkkeys(keycols::Vector{Symbol}, c::DataFrame, side::String)
    for k in keycols
        String(k) in names(c) || throw(ArgumentError(
            "futurejoin key column $k not found in $side input"))
    end
    return nothing
end

# Pull the next right chunk (type-unstable, once per right chunk): create or
# widen the store when the promoted right schema moves. The matches vector and
# the per-key buffers may be half-consumed mid-left-chunk when this runs, so
# they are converted along with the store's key and value types.
function pullright!(js::FutureJoinState, cfg::FutureJoinConfig)
    next = js.rstarted ? iterate(js.rchunks, js.rstate) : iterate(js.rchunks)
    js.rstarted = true
    if next === nothing
        js.rdone = true
        js.rnt === nothing && (js.passthrough = true)
        return nothing
    end
    chunk, js.rstate = next
    if js.rvaluenames === nothing
        checkkeys(cfg.keycols, chunk, "right")
        js.rvaluenames = Symbol[n for n in propertynames(chunk)
                     if n !== :time && !(n in cfg.keycols)]
    end
    types = promotetypes(js.rtypes, chunktypes(chunk))
    widened = js.rtypes !== nothing && types != js.rtypes
    js.rtypes = types
    js.rnt = Tables.columntable(chunk)
    js.rpos = 1
    if js.store === nothing
        V = storerowtype(types)
        K = storekeytype(types, cfg.keynames)
        js.store = Dict{K,KeyBuffer{V}}()
        js.matches = Union{Missing,V}[]
    elseif widened
        K = storekeytype(types, cfg.keynames)
        V = storerowtype(types)
        js.store = Dict{K,KeyBuffer{V}}(
            convert(K, k) => KeyBuffer{V}(convert(Vector{V}, b.rows), b.head)
            for (k, b) in js.store)
        js.matches = convert(Vector{Union{Missing,V}}, js.matches)
    end
    return nothing
end

function joinchunk!(js::FutureJoinState, cfg::FutureJoinConfig, c::DataFrame)
    if !js.leftchecked
        checkkeys(cfg.keycols, c, "left")
        js.leftchecked = true
    end
    js.rnt === nothing && !js.rdone && pullright!(js, cfg)
    js.passthrough && return prefixleft!(cfg, c)
    if !js.checked
        checknames(cfg, c, js.rvaluenames)
        js.checked = true
    end
    nt = Tables.columntable(c)
    resize!(js.matches, nrow(c))
    fill!(js.matches, missing)   # leave no undef slots for a mid-chunk widen
    i = 1
    while true
        i, js.rpos, needpull = futuresegment!(js.matches, js.store, nt, i,
            js.rnt, js.rpos, js.rdone,
            cfg.keynames, cfg.after,
            cfg.tolerance)
        needpull || break
        pullright!(js, cfg)
    end
    return assemble(cfg, js, prefixleft!(cfg, c))
end

# --- merge kernel ----------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access. On entry it drains the current right chunk fully into
# the per-key buffers (a forward join needs rows *ahead* of t, so admission is
# not gated by t), then processes left rows from index i. Returns
# (i, rpos, needpull): needpull means a left row's key has no buffered future
# row yet but the right stream is not exhausted — the driver must pull the
# next right chunk before row i can be resolved.
function futuresegment!(matches::Vector{Union{Missing,V}},
    store::Dict{K,KeyBuffer{V}}, lnt::NamedTuple, i::Int, rnt::NamedTuple,
    rpos::Int, rdone::Bool, keynames::Val{KN}, after::A,
    tolerance) where {K,V,KN,A}
    rlen = length(rnt.time)
    while rpos <= rlen
        b = get!(() -> KeyBuffer{V}(), store, keyat(rnt, rpos, keynames))
        bufpush!(b, rowat(V, rnt, rpos))
        rpos += 1
    end
    n = length(lnt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        b = get(store, keyat(lnt, i, keynames), nothing)
        # Drop buffered rows now before t (strict: at or before t); they can
        # never match this or any later (larger-t) left row. `after` is `>`
        # or `>=`, so its negation is the discard test — the reverse of
        # asofjoin's `before`, never reused.
        if b !== nothing
            while !bufempty(b) && !after(buffront(b).time, t)
                bufpop!(b)
            end
        end
        if b === nothing || bufempty(b)
            # No buffered future row for this key. If the right stream may
            # still hold one, pull and re-enter at i; otherwise it is missing.
            rdone || return (i, rpos, true)
        else
            # The front is the earliest right row with time >= t (strict >).
            # Tolerance staleness is decided here, against each left row: a
            # front too far ahead now may match a later, larger t, so never
            # evict on tolerance — leave the slot missing.
            m = buffront(b)
            if tolerance === nothing || m.time - t <= tolerance
                @inbounds matches[i] = m
            end
        end
        i += 1
    end
    return (i, rpos, false)
end

# --- output assembly -------------------------------------------------------

function prefixleft!(cfg::FutureJoinConfig, c::DataFrame)
    cfg.leftprefix === nothing && return c
    for n in propertynames(c)
        (n === :time || n in cfg.keycols) && continue
        rename!(c, n => prefixed(cfg.leftprefix, n))
    end
    return c
end

# Needs both schemas, so it runs once the first right chunk has been seen and
# before the first chunk is emitted.
function checknames(cfg::FutureJoinConfig, c::DataFrame, rvaluenames)
    seen = Set{Symbol}()
    function check(n)
        n in seen && throw(
            ArgumentError(
                "futurejoin output column $n appears more than once; use leftprefix/rightprefix to disambiguate",
            ),
        )
        push!(seen, n)
        return nothing
    end
    check(:time)
    foreach(check, cfg.keycols)
    for n in propertynames(c)
        (n === :time || n in cfg.keycols) && continue
        check(prefixed(cfg.leftprefix, n))
    end
    for n in rvaluenames
        check(prefixed(cfg.rightprefix, n))
    end
    cfg.righttime === nothing || check(cfg.righttime)
    return nothing
end

function assemble(cfg::FutureJoinConfig, js::FutureJoinState, c::DataFrame)
    rdf = DataFrame()
    for n in js.rvaluenames
        rdf[!, prefixed(cfg.rightprefix, n)] = matchcolumn(js.matches, Val(n))
    end
    cfg.righttime === nothing ||
        (rdf[!, cfg.righttime] = matchcolumn(js.matches, Val(:time)))
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, rdf; copycols = false)
end

end # module Acausal
