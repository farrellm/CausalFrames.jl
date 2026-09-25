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
    asofjoin(left::CausalPipeline, right::CausalPipeline; ...) -> CausalPipeline

A transform joining each left row to the latest `right` row at or before its
time. Every left row is kept, with `right`'s non-time columns appended as
`Union{Missing, T}`: `missing` where no right row matches. Among right rows at
the same time, the last one wins. For a table with no time column, use
[`lookupjoin`](@ref).

# Arguments
- `right`: the pipeline to join from. If it produces no rows, left rows pass
  through unchanged (apart from `leftprefix`).

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present on both sides; a row matches only right rows with an
  equal key. Key columns appear once, from the left, never prefixed.
- `tolerance = nothing`: the maximum age of a match, `time - righttime <=
  tolerance`; must be non-negative. The right pipeline then runs over
  `[start - tolerance, stop)`, so rows near `start` can match earlier right
  rows; the time type must support subtraction. Without it, right runs over
  `[start, stop)` only.
- `strict = false`: match only right rows strictly before the left row.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique, so a self
  join needs a prefix.
- `righttime = nothing`: a name under which to keep the matched right row's
  time; by default it is dropped.
"""
function asofjoin(right::CausalPipeline; key = nothing, tolerance = nothing,
    strict::Bool = false, leftprefix = nothing,
    rightprefix = nothing,
    righttime::Union{Nothing,Symbol} = nothing)
    keycols = keycolumns(key, "asofjoin")
    righttime === :time && throw(
        ArgumentError(
            "asofjoin righttime may not be :time; it would collide with the left time column",
        ),
    )
    righttime !== nothing && righttime in keycols &&
        throw(
            ArgumentError(
                "asofjoin righttime $(repr(righttime)) collides with a key column"),
        )
    lp = normprefix(leftprefix)
    rp = normprefix(rightprefix)
    return function (left::CausalPipeline)
        return CausalPipeline() do ctx::Context
            cfg = JoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                strict ? (<) : (<=), Backward(), lp, rp, righttime, "asofjoin")
            js = JoinState(right.run(rightcontext(ctx, tolerance)))
            return chunkmap(c -> joinchunk!(js, cfg, c), left.run(ctx))
        end
    end
end
asofjoin(left::CausalPipeline, right::CausalPipeline; kwargs...) =
    asofjoin(right; kwargs...)(left)

normprefix(::Nothing) = nothing
normprefix(p::Union{Symbol,AbstractString}) = String(p)

prefixed(::Nothing, n::Symbol) = n
prefixed(p::String, n::Symbol) = Symbol(p, '_', n)

# `op` names the operator in the error: applymodels widens its models' context
# through here too.
rightcontext(ctx::Context, ::Nothing, op::String = "asofjoin") = ctx
function rightcontext(ctx::Context, tolerance, op::String = "asofjoin")
    start = ctx.start - tolerance
    start <= ctx.start || throw(ArgumentError(
        "$op tolerance must be non-negative, got $tolerance"))
    return Context(start, ctx.stop)
end

# The join engine shared by `asofjoin`, `applymodels` and the acausal
# `futurejoin`. The direction `D` is a singleton type picking the store and its
# kernel by dispatch: `Backward` here (the latest right row at or before each
# left row), `Acausal.Forward` in the submodule. strict and tolerance ride in
# type parameters (`cmp` is `<`/`<=` backward, `>`/`>=` forward), so the kernel
# specializes and neither costs a per-row branch. `op` names the operator in
# error messages.
struct Backward end

struct JoinConfig{KN,Tol,C,D}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    tolerance::Tol
    cmp::C
    direction::D
    leftprefix::Union{Nothing,String}
    rightprefix::Union{Nothing,String}
    righttime::Union{Nothing,Symbol}
    op::String
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the `segment!` function barrier.
mutable struct JoinState
    const right::PullCursor  # the right chunks; `done` once exhausted
    rnt::Any           # current right column table (nothing until first pull)
    rpos::Int          # index of the next unadmitted right row in rnt
    rvaluenames::Any   # Vector{Symbol}: right columns minus keys minus time
    rtypes::Union{Nothing,NamedTuple}  # promotion of right schemas seen
    store::Any         # the direction's store (`newstore`), keyed by K over rows V
    matches::Any       # Vector{V}: per-left-row match, reused; see `found`
    found::Vector{Bool} # which matches slots hold a match; the rest are undef
    passthrough::Bool  # the right stream produced no chunks at all
    leftchecked::Bool  # key validation against the left schema done
    checked::Bool      # output-name duplicate validation done
    JoinState(rchunks) = new(PullCursor(rchunks), nothing, 1, nothing,
        nothing, nothing, nothing, Bool[], false, false, false)
end

# The backward store, shared with `lastrow`: a `Dict{K,Int}` of slot numbers
# over a `Vector{V}` of rows rather than a Dict of rows, because a Dict of rows
# can only answer a lookup as `Union{Nothing,V}` — and building that Union out
# of an inline-stored V heap-allocates it whenever V is not an isbits type (any
# String or Missing-admitting column is enough). An Int slot number is isbits,
# so the lookup is free and the row is read back inline.
struct SlotStore{K,V}
    index::Dict{K,Int}
    slots::Vector{V}
end
SlotStore{K,V}() where {K,V} = SlotStore{K,V}(Dict{K,Int}(), V[])

newstore(::Backward, ::Type{K}, ::Type{V}) where {K,V} = SlotStore{K,V}()

# Slot numbers do not move, so only the keys are rebuilt; every slot is
# occupied, so that vector converts wholesale.
widenstore(st::SlotStore, ::Type{K}, ::Type{V}) where {K,V} =
    SlotStore{K,V}(Dict{K,Int}(convert(K, k) => j for (k, j) in st.index),
        convert(Vector{V}, st.slots))

# Store `row` as key `k`'s latest. `get!` claims the next slot number on a
# miss, so this costs one hash whether or not the key is new.
@inline function admitslot!(index::Dict{K,Int}, slots::Vector{V}, k,
    row::V) where {K,V}
    j = get!(index, k, length(slots) + 1)
    j > length(slots) ? push!(slots, row) : (@inbounds slots[j] = row)
    return nothing
end

# The concrete row and key NamedTuple types for the store, from the promoted
# right schema. The key type comes from the right side; left-side lookups may
# carry different (say, narrower numeric) value types — Dict lookup hashes
# with isequal, which matches across numeric types, so no conversion needed.
storerowtype(types::NamedTuple) = NamedTuple{keys(types),Tuple{values(types)...}}
storekeytype(types::NamedTuple, ::Val{KN}) where {KN} =
    storerowtype(NamedTuple{KN}(types))

# Widen a half-filled matches buffer, copying only the slots `found` marks.
# The rest are deliberately undefined — a plain `convert` over the vector would
# read them. Shared with futurejoin, whose buffer has the same shape.
function convertmatches(::Type{V2}, matches::Vector,
    found::Vector{Bool}) where {V2}
    out = Vector{V2}(undef, length(matches))
    @inbounds for i in eachindex(matches)
        found[i] && (out[i] = convert(V2, matches[i]))
    end
    return out
end

# Pull the next right chunk (type-unstable, once per right chunk): create or
# widen the store when the promoted right schema moves — a source may hand a
# column a different element type from one chunk to the next. The matches
# buffer may be half-filled mid-left-chunk when this runs, so it is converted
# along with the store — through `convertmatches`, since its unmatched slots
# are undefined.
function pullright!(js::JoinState, cfg::JoinConfig)
    chunk = pull!(js.right)
    if chunk === nothing
        js.rnt === nothing && (js.passthrough = true)
        return nothing
    end
    if js.rvaluenames === nothing
        checkkeycolumns(cfg.keycols, chunk, cfg.op, "the right input")
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
        js.store = newstore(cfg.direction, storekeytype(types, cfg.keynames), V)
        js.matches = V[]
    elseif widened
        V = storerowtype(types)
        js.store = widenstore(js.store, storekeytype(types, cfg.keynames), V)
        js.matches = convertmatches(V, js.matches, js.found)
    end
    return nothing
end

function joinchunk!(js::JoinState, cfg::JoinConfig, c::DataFrame)
    if !js.leftchecked
        checkkeycolumns(cfg.keycols, c, cfg.op, "the left input")
        js.leftchecked = true
    end
    js.rnt === nothing && !js.right.done && pullright!(js, cfg)
    js.passthrough && return prefixleft!(cfg, c)
    if !js.checked
        checknames(cfg, c, js.rvaluenames)
        js.checked = true
    end
    matchchunk!(js, cfg, Tables.columntable(c))
    return assemble(cfg, js, prefixleft!(cfg, c))
end

# Match every row of the left column table `nt` into `js.matches`/`js.found`,
# pulling right chunks as the kernel asks for them. The fields are re-read on
# each pass: a widening inside `pullright!` replaces the store and the buffer.
function matchchunk!(js::JoinState, cfg::JoinConfig, nt::NamedTuple)
    n = length(nt.time)
    resize!(js.matches, n)
    resize!(js.found, n)
    fill!(js.found, false)   # the only reset needed; matches slots are guarded
    i = 1
    while true
        i, js.rpos, needpull = segment!(js.store, js.matches, js.found, nt, i,
            js.rnt, js.rpos, js.right.done, cfg.keynames, cfg.cmp,
            cfg.tolerance)
        needpull || break
        pullright!(js, cfg)
    end
    return nothing
end

# --- merge kernel ----------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access. Processes left rows from index i, admitting right
# rows from rnt starting at rpos into the store. Returns (i, rpos, needpull):
# needpull means the current right chunk is consumed but the stream may still
# hold rows admissible for left row i — the driver must pull the next right
# chunk before row i can be matched.
segment!(st::SlotStore{K,V}, matches::Vector{V}, found::Vector{Bool},
    lnt::NamedTuple, i::Int, rnt::NamedTuple, rpos::Int, rdone::Bool,
    keynames::Val, before, tolerance) where {K,V} =
    joinsegment!(matches, found, st.index, st.slots, lnt, i, rnt, rpos, rdone,
        keynames, before, tolerance)

function joinsegment!(matches::Vector{V}, found::Vector{Bool},
    index::Dict{K,Int}, slots::Vector{V}, lnt::NamedTuple,
    i::Int, rnt::NamedTuple, rpos::Int, rdone::Bool,
    keynames::Val{KN}, before::B, tolerance) where {K,V,KN,B}
    n = length(lnt.time)
    rlen = length(rnt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        # Admit right rows not after (strict: strictly before) t; equal right
        # times overwrite the key's slot, so the later row in stream order
        # wins.
        while rpos <= rlen && before(@inbounds(rnt.time[rpos]), t)
            admitslot!(index, slots, keyat(rnt, rpos, keynames),
                rowat(V, rnt, rpos))
            rpos += 1
        end
        rpos > rlen && !rdone && return (i, rpos, true)
        # A stored row keeps its time, so tolerance staleness is decided here,
        # against each left row — never by evicting eagerly.
        j = get(index, keyat(lnt, i, keynames), 0)
        if j > 0
            m = @inbounds slots[j]
            if tolerance === nothing || t - m.time <= tolerance
                @inbounds matches[i] = m
                @inbounds found[i] = true
            end
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

# One rename! over every pair, not one call per column: each call rebuilds the
# chunk's column index, which made this O(ncols^2) per chunk. The field form is
# shared with lookupjoin.
prefixleft!(cfg::JoinConfig, c::DataFrame) =
    prefixleft!(cfg.leftprefix, cfg.keycols, c)
function prefixleft!(leftprefix::Union{Nothing,String}, keycols::Vector{Symbol},
    c::DataFrame)
    leftprefix === nothing && return c
    pairs = [
        n => prefixed(leftprefix, n) for n in propertynames(c)
        if n !== :time && !(n in keycols)
    ]
    isempty(pairs) || rename!(c, pairs)
    return c
end

# Needs both schemas, so it runs once the first right chunk has been seen and
# before the first chunk is emitted.
function checknames(cfg::JoinConfig, c::DataFrame, rvaluenames)
    seen = Set{Symbol}()
    function check(n)
        n in seen && throw(
            ArgumentError(
                "$(cfg.op) output column $(repr(n)) appears more than once; use `leftprefix`/`rightprefix` to disambiguate",
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

function assemble(cfg::JoinConfig, js::JoinState, c::DataFrame)
    rdf = DataFrame()
    for n in js.rvaluenames
        rdf[!, prefixed(cfg.rightprefix, n)] =
            matchcolumn(js.matches, js.found, Val(n))
    end
    cfg.righttime === nothing ||
        (rdf[!, cfg.righttime] = matchcolumn(js.matches, js.found, Val(:time)))
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, rdf; copycols = false)
end

# Function barrier: fieldtype fixes the column's element type so the
# comprehension builds a typed column directly. `found` is what makes the
# unmatched slots safe — they are undefined, never `missing`, because a
# Vector{Union{Missing,V}} boxes every stored row when V is not isbits.
matchcolumn(matches::Vector{V}, found::Vector{Bool}, ::Val{N}) where {V,N} =
    Union{Missing,fieldtype(V, N)}[
        @inbounds(found[i]) ? getproperty(@inbounds(matches[i]), N) : missing
        for i in eachindex(matches)]
