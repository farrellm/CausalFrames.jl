# The as-of join transform. The left stream drives a chunkmap; the right
# stream is pulled on demand from inside the step — a two-pointer merge where
# the right pointer advances per left row, not per chunk. Per the summarize.jl
# conventions, the type-unstable setup (schemas, building or widening the
# store) happens once per right chunk, and the merge kernel takes concretely
# typed arguments behind a function barrier.

"""
    asofjoin(right::CausalPipeline; key = nothing, tolerance = nothing,
             strict = false, leftprefix = nothing, rightprefix = nothing,
             righttime = nothing) -> (CausalPipeline -> CausalPipeline)

A transform joining each left row to the most recent right row whose time is
not after the left row's time (`strict = true`: strictly before). Every left
row is kept; the right table's value columns are appended with element type
`Union{Missing, T}` and are `missing` where no right row qualifies. Among
right rows sharing one time, the last in stream order wins. The right `time`
column is dropped unless `righttime` names an output column for the matched
row's time.

With `key` (a column name or collection of column names, present in both
tables) rows join per unique key value, exact-matched; the key columns
appear once in the output, taken from the left row, never prefixed. With
`tolerance` the match additionally requires `time - rtime <= tolerance`, and
the right pipeline runs over the widened context `[start - tolerance, stop)`
so lookback near the window start is fully covered — this requires the time
type to support subtraction (numbers and `Dates` types do). Without
`tolerance` the right pipeline sees only `[start, stop)`, so left rows near
`start` may find no earlier right row.

`leftprefix` / `rightprefix` rename that side's non-time, non-key columns to
`"{prefix}_{name}"`. Output column names must be unique after prefixing —
in particular a self join (`p |> asofjoin(p)`) needs a prefix. A right
stream producing no chunks passes left chunks through unchanged (no right
columns, no `righttime`), except for the `leftprefix` rename.
"""
function asofjoin(right::CausalPipeline; key = nothing, tolerance = nothing,
                  strict::Bool = false, leftprefix = nothing,
                  rightprefix = nothing,
                  righttime::Union{Nothing,Symbol} = nothing)
    keycols = tokeycolumns(key)
    allunique(keycols) ||
        throw(ArgumentError("asofjoin key columns must be unique"))
    :time in keycols && throw(ArgumentError(
        "time is the as-of dimension and may not be an asofjoin key"))
    righttime === :time && throw(ArgumentError(
        "asofjoin righttime may not be time; it would collide with the left time column"))
    righttime !== nothing && righttime in keycols && throw(ArgumentError(
        "asofjoin righttime $righttime collides with a key column"))
    lp = normprefix(leftprefix)
    rp = normprefix(rightprefix)
    return function (left::CausalPipeline)
        return CausalPipeline() do ctx::Context
            cfg = AsofJoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                                 strict ? (<) : (<=), lp, rp, righttime)
            js = AsofJoinState(right.run(rightcontext(ctx, tolerance)))
            return chunkmap(c -> joinchunk!(js, cfg, c), left.run(ctx))
        end
    end
end

normprefix(::Nothing) = nothing
normprefix(p::Union{Symbol,AbstractString}) = String(p)

prefixed(::Nothing, n::Symbol) = n
prefixed(p::String, n::Symbol) = Symbol(p, '_', n)

rightcontext(ctx::Context, ::Nothing) = ctx
function rightcontext(ctx::Context, tolerance)
    start = ctx.start - tolerance
    start <= ctx.start || throw(ArgumentError(
        "asofjoin tolerance must be non-negative, got $tolerance"))
    return Context(start, ctx.stop)
end

# strict and tolerance ride in type parameters (`before` is `<` or `<=`), so
# the kernel specializes and neither costs a per-row branch.
struct AsofJoinConfig{KN,Tol,B}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    tolerance::Tol
    before::B
    leftprefix::Union{Nothing,String}
    rightprefix::Union{Nothing,String}
    righttime::Union{Nothing,Symbol}
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the joinsegment! function barrier.
mutable struct AsofJoinState
    rchunks::Any       # right chunk iterator
    rstate::Any        # its iteration state
    rstarted::Bool
    rdone::Bool        # right stream exhausted
    rnt::Any           # current right column table (nothing until first pull)
    rpos::Int          # index of the next unadmitted right row in rnt
    rvaluenames::Any   # Vector{Symbol}: right columns minus keys minus time
    rtypes::Union{Nothing,NamedTuple}  # promotion of right schemas seen
    store::Any         # Dict{K,V}: per-key most recent admitted right row
    matches::Any       # Vector{Union{Missing,V}}: per-left-row match, reused
    passthrough::Bool  # the right stream produced no chunks at all
    leftchecked::Bool  # key validation against the left schema done
    checked::Bool      # output-name duplicate validation done
    AsofJoinState(rchunks) = new(rchunks, nothing, false, false, nothing, 1,
                                 nothing, nothing, nothing, nothing, false,
                                 false, false)
end

# The concrete row and key NamedTuple types for the store, from the promoted
# right schema. The key type comes from the right side; left-side lookups may
# carry different (say, narrower numeric) value types — Dict lookup hashes
# with isequal, which matches across numeric types, so no conversion needed.
storerowtype(types::NamedTuple) = NamedTuple{keys(types),Tuple{values(types)...}}
storekeytype(types::NamedTuple, ::Val{KN}) where {KN} =
    storerowtype(NamedTuple{KN}(types))

function checkkeys(keycols::Vector{Symbol}, c::DataFrame, side::String)
    for k in keycols
        String(k) in names(c) || throw(ArgumentError(
            "asofjoin key column $k not found in $side input"))
    end
    return nothing
end

# Pull the next right chunk (type-unstable, once per right chunk): create or
# widen the store when the promoted right schema moves — readcsv infers
# column types per chunk. The matches vector may be half-filled mid-left-chunk
# when this runs, so it is converted along with the store.
function pullright!(js::AsofJoinState, cfg::AsofJoinConfig)
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
        js.rvaluenames = Symbol[n for n in Symbol.(names(chunk))
                                if n !== :time && !(n in cfg.keycols)]
    end
    types = promotetypes(js.rtypes, chunktypes(chunk))
    widened = js.rtypes !== nothing && types != js.rtypes
    js.rtypes = types
    js.rnt = Tables.columntable(chunk)
    js.rpos = 1
    if js.store === nothing
        V = storerowtype(types)
        js.store = Dict{storekeytype(types, cfg.keynames),V}()
        js.matches = Union{Missing,V}[]
    elseif widened
        K = storekeytype(types, cfg.keynames)
        V = storerowtype(types)
        js.store = Dict{K,V}(convert(K, k) => convert(V, v)
                             for (k, v) in js.store)
        js.matches = convert(Vector{Union{Missing,V}}, js.matches)
    end
    return nothing
end

function joinchunk!(js::AsofJoinState, cfg::AsofJoinConfig, c::DataFrame)
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
        i, js.rpos, needpull = joinsegment!(js.matches, js.store, nt, i,
                                            js.rnt, js.rpos, js.rdone,
                                            cfg.keynames, cfg.before,
                                            cfg.tolerance)
        needpull || break
        pullright!(js, cfg)
    end
    return assemble(cfg, js, prefixleft!(cfg, c))
end

# --- merge kernel ----------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access. Processes left rows from index i, admitting right
# rows from rnt starting at rpos into the store. Returns (i, rpos, needpull):
# needpull means the current right chunk is consumed but the stream may still
# hold rows admissible for left row i — the driver must pull the next right
# chunk before row i can be matched.
function joinsegment!(matches::Vector{Union{Missing,V}}, store::Dict{K,V},
                      lnt::NamedTuple, i::Int, rnt::NamedTuple, rpos::Int,
                      rdone::Bool, keynames::Val{KN}, before::B,
                      tolerance) where {K,V,KN,B}
    n = length(lnt.time)
    rlen = length(rnt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        # Admit right rows not after (strict: strictly before) t; equal right
        # times overwrite the store, so the later row in stream order wins.
        while rpos <= rlen && before(@inbounds(rnt.time[rpos]), t)
            store[keyat(rnt, rpos, keynames)] = rowat(V, rnt, rpos)
            rpos += 1
        end
        rpos > rlen && !rdone && return (i, rpos, true)
        # A stored row keeps its time, so tolerance staleness is decided here,
        # against each left row — never by evicting eagerly.
        m = get(store, keyat(lnt, i, keynames), nothing)
        if m !== nothing && (tolerance === nothing || t - m.time <= tolerance)
            @inbounds matches[i] = m
        end
        i += 1
    end
    return (i, rpos, false)
end

# The V(...) conversion is what keeps the store insert type-stable when a
# column's eltype is abstract (e.g. Union{Missing,Int}): a bare map would
# yield the values' narrower concrete types.
@inline rowat(::Type{V}, nt::NamedTuple, i::Int) where {V} =
    convert(V, map(col -> @inbounds(col[i]), nt))
@inline keyat(nt::NamedTuple, i::Int, ::Val{KN}) where {KN} =
    NamedTuple{KN}(map(c -> @inbounds(getproperty(nt, c)[i]), KN))

# --- output assembly -------------------------------------------------------

function prefixleft!(cfg::AsofJoinConfig, c::DataFrame)
    cfg.leftprefix === nothing && return c
    for n in Symbol.(names(c))
        (n === :time || n in cfg.keycols) && continue
        rename!(c, n => prefixed(cfg.leftprefix, n))
    end
    return c
end

# Needs both schemas, so it runs once the first right chunk has been seen and
# before the first chunk is emitted.
function checknames(cfg::AsofJoinConfig, c::DataFrame, rvaluenames)
    seen = Set{Symbol}()
    check(n) = n in seen ? throw(ArgumentError(
        "asofjoin output column $n appears more than once; use leftprefix/rightprefix to disambiguate")) :
        push!(seen, n)
    check(:time)
    foreach(check, cfg.keycols)
    for n in Symbol.(names(c))
        (n === :time || n in cfg.keycols) && continue
        check(prefixed(cfg.leftprefix, n))
    end
    for n in rvaluenames
        check(prefixed(cfg.rightprefix, n))
    end
    cfg.righttime === nothing || check(cfg.righttime)
    return nothing
end

function assemble(cfg::AsofJoinConfig, js::AsofJoinState, c::DataFrame)
    rdf = DataFrame()
    for n in js.rvaluenames
        rdf[!, prefixed(cfg.rightprefix, n)] = matchcolumn(js.matches, Val(n))
    end
    cfg.righttime === nothing ||
        (rdf[!, cfg.righttime] = matchcolumn(js.matches, Val(:time)))
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, rdf; copycols = false)
end

# Function barrier: fieldtype fixes the column's element type so the
# comprehension builds a typed column directly. The eltype is taken apart
# with nonmissingtype rather than matched as Union{Missing,V}, which would
# leave V unbound.
matchcolumn(matches::Vector{T}, ::Val{N}) where {T,N} =
    Union{Missing,fieldtype(nonmissingtype(T), N)}[
        m === missing ? missing : getproperty(m, N) for m in matches]
