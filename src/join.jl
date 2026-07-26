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
    asofjoin(left::CausalPipeline, right::CausalPipeline; key = nothing,
             ...) -> CausalPipeline

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

The curried form composes with `|>`; the uncurried form applies directly, so
`asofjoin(left, right; ...)` is equivalent to `left |> asofjoin(right; ...)`.
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
    righttime === :time && throw(
        ArgumentError(
            "asofjoin righttime may not be time; it would collide with the left time column",
        ),
    )
    righttime !== nothing && righttime in keycols &&
        throw(ArgumentError(
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
asofjoin(left::CausalPipeline, right::CausalPipeline; kwargs...) =
    asofjoin(right; kwargs...)(left)

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
    index::Any         # Dict{K,Int}: per-key slot holding its most recent row
    slots::Any         # Vector{V}: those rows, one slot per key ever seen
    matches::Any       # Vector{V}: per-left-row match, reused; see `found`
    found::Vector{Bool} # which matches slots hold a match; the rest are undef
    passthrough::Bool  # the right stream produced no chunks at all
    leftchecked::Bool  # key validation against the left schema done
    checked::Bool      # output-name duplicate validation done
    AsofJoinState(rchunks) = new(rchunks, nothing, false, false, nothing, 1,
        nothing, nothing, nothing, nothing, nothing,
        Bool[], false, false, false)
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

function checkkeys(keycols::Vector{Symbol}, c::DataFrame, side::String)
    for k in keycols
        String(k) in names(c) || throw(ArgumentError(
            "asofjoin key column $k not found in $side input"))
    end
    return nothing
end

# Pull the next right chunk (type-unstable, once per right chunk): create or
# widen the store when the promoted right schema moves — a source may hand a
# column a different element type from one chunk to the next. The matches
# buffer may be half-filled mid-left-chunk when this runs, so it is converted
# along with the store — through `convertmatches`, since its unmatched slots
# are undefined.
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
        js.rvaluenames = Symbol[n for n in propertynames(chunk)
                     if n !== :time && !(n in cfg.keycols)]
    end
    types = promotetypes(js.rtypes, chunktypes(chunk))
    widened = js.rtypes !== nothing && types != js.rtypes
    js.rtypes = types
    js.rnt = Tables.columntable(chunk)
    js.rpos = 1
    if js.index === nothing
        V = storerowtype(types)
        js.index = Dict{storekeytype(types, cfg.keynames),Int}()
        js.slots = V[]
        js.matches = V[]
    elseif widened
        K = storekeytype(types, cfg.keynames)
        V = storerowtype(types)
        # Slot numbers do not move, so only the keys are rebuilt; every slot is
        # occupied, so that vector converts wholesale.
        js.index = Dict{K,Int}(convert(K, k) => j for (k, j) in js.index)
        js.slots = convert(Vector{V}, js.slots)
        js.matches = convertmatches(V, js.matches, js.found)
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
    resize!(js.found, nrow(c))
    fill!(js.found, false)   # the only reset needed; matches slots are guarded
    i = 1
    while true
        i, js.rpos, needpull = joinsegment!(js.matches, js.found, js.index,
            js.slots, nt, i, js.rnt, js.rpos,
            js.rdone, cfg.keynames, cfg.before,
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
# The store is an index and a slot vector rather than a Dict of rows, because a
# Dict of rows can only answer a lookup as `Union{Nothing,V}` — and building
# that Union out of an inline-stored V heap-allocates it, once per left row,
# whenever V is not an isbits type (any String or Missing-admitting right
# column is enough). An Int slot number is isbits, so the lookup is free and
# the row is read back inline.
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
        # wins. `get!` claims the next slot number on a miss, so admitting a
        # row costs one hash whether or not the key is new.
        while rpos <= rlen && before(@inbounds(rnt.time[rpos]), t)
            row = rowat(V, rnt, rpos)
            j = get!(index, keyat(rnt, rpos, keynames), length(slots) + 1)
            j > length(slots) ? push!(slots, row) : (@inbounds slots[j] = row)
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
# chunk's column index, which made this O(ncols^2) per chunk.
function prefixleft!(cfg::AsofJoinConfig, c::DataFrame)
    cfg.leftprefix === nothing && return c
    pairs = [
        n => prefixed(cfg.leftprefix, n) for n in propertynames(c)
        if n !== :time && !(n in cfg.keycols)
    ]
    isempty(pairs) || rename!(c, pairs)
    return c
end

# Needs both schemas, so it runs once the first right chunk has been seen and
# before the first chunk is emitted.
function checknames(cfg::AsofJoinConfig, c::DataFrame, rvaluenames)
    seen = Set{Symbol}()
    function check(n)
        n in seen && throw(
            ArgumentError(
                "asofjoin output column $n appears more than once; use leftprefix/rightprefix to disambiguate",
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

function assemble(cfg::AsofJoinConfig, js::AsofJoinState, c::DataFrame)
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
