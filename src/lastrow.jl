# The last-row-per-key transform. It folds the whole window the way `summarize`
# does — one pass per chunk, nothing emitted until the stream is exhausted — but
# keeps rows rather than summaries, so its store is join.jl's rather than a
# GroupTable: a Dict{K,Int} of slot numbers over a Vector{V} of concretely typed
# rows. A Dict{K,V} would answer every lookup as Union{Nothing,V} and box it
# whenever V is not isbits (one String column is enough), and a Dict{K,DataFrame}
# of one-row slices would allocate a whole DataFrame index per key. See
# DESIGN.md, "Representing a match". No `found` mask is needed here, unlike the
# join's matches buffer: every slot a key claims is written the same instant, so
# the store is never half-filled and widening is a plain `convert`.

"""
    lastrow(; key = nothing) -> (CausalPipeline -> CausalPipeline)
    lastrow(p::CausalPipeline; key = nothing) -> CausalPipeline

A transform emitting the stream's **last row**, retimed to the context's end
time `stop`. Every input column is kept, carrying its value from that row: the
output schema is the input's exactly, `:time` included and in its own position.

Without `key` the output is exactly one row, and an input with no rows produces
no output at all — unlike keyless [`summarize`](@ref), which has an identity
summary to emit. With `key` (a column name or collection of column names) one
row is emitted per distinct key value, each being that key's last row, all
retimed to `stop` and emitted sorted by key — legal precisely because every
emitted time is then equal.

The original timestamp is **lost**: `:time` is overwritten with `stop`. Carry it
forward under another name if you need it:

```julia
p |> addcolumns(r -> (; t0 = r.time)) |> lastrow(; key = :sym)
```

`time` may not be a key and key columns must be unique — both `ArgumentError`s
when the transform is built — and the key columns must be present in the input,
checked on its first chunk. A chunk whose column names differ from the first
chunk's, in content or in order, is an `ArgumentError`; element *types* may
differ from chunk to chunk, as everywhere else, and the store tracks their
promotion.

Like [`summarize`](@ref), `lastrow` is causal — a row emitted at `stop` folds
only rows with time `<= stop` — and stateful over the whole window, emitting
once when its input is exhausted, so its stream is a single frame over
`[start, stop]`.

The curried form composes with `|>`; the uncurried form applies directly, so
`lastrow(p; key)` is equivalent to `p |> lastrow(; key)`.
"""
function lastrow(; key = nothing)
    keycols = tokeycolumns(key)
    allunique(keycols) || throw(ArgumentError("lastrow key columns must be unique"))
    :time in keycols && throw(
        ArgumentError("time is the ordering dimension and may not be a lastrow key"),
    )
    keynames = Val(Tuple(keycols))
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            st = LastRowState(keycols)
            step = function (c)
                lastrowchunk!(st, keynames, c)
                return nothing
            end
            return chunkmap(step, p.run(ctx);
                flush = () -> flushlastrow(st, keynames, ctx.stop))
        end
    end
end
lastrow(p::CausalPipeline; kwargs...) = lastrow(; kwargs...)(p)

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state: the
# store's row and key types come from the promoted input schema, so they are not
# known until the first chunk arrives, and everything per-row sits behind the
# `lastsegment!` function barrier.
mutable struct LastRowState
    const keycols::Vector{Symbol}
    names::Union{Nothing,Vector{String}}  # column names, fixed by the first chunk
    types::Union{Nothing,NamedTuple}      # promotion of every input schema seen
    index::Any    # Dict{K,Int}: per-key slot holding that key's last row
    slots::Any    # Vector{V}: those rows, one slot per key ever seen
    row::Any      # keyless: the last row seen, as a V
end
LastRowState(keycols::Vector{Symbol}) =
    LastRowState(keycols, nothing, nothing, nothing, nothing, nothing)

function lastrowchunk!(st::LastRowState, ::Val{KN}, c::DataFrame) where {KN}
    checkschema!(st, c)
    types = promotetypes(st.types, chunktypes(c))
    widened = st.types !== nothing && types != st.types
    st.types = types
    V = storerowtype(types)
    nt = Tables.columntable(c)
    if isempty(KN)
        # Keyless: the chunk's last row is the last row so far, so there is
        # nothing to do per row. O(ncols) per chunk.
        st.row = rowat(V, nt, length(nt.time))
        return nothing
    end
    if st.index === nothing
        st.index = Dict{storekeytype(types, Val(KN)),Int}()
        st.slots = V[]
    elseif widened
        # Slot numbers do not move, so only the keys are rebuilt; every slot is
        # occupied, so that vector converts wholesale (join.jl's pullright!).
        K = storekeytype(types, Val(KN))
        st.index = Dict{K,Int}(convert(K, k) => j for (k, j) in st.index)
        st.slots = convert(Vector{V}, st.slots)
    end
    lastsegment!(st.index, st.slots, nt, Val(KN))
    return nothing
end

# O(ncols) per chunk. The key check needs the schema, so it runs on the first
# chunk; the names check runs on every chunk, since the store's row type is fixed
# by them and a rename would otherwise surface as an opaque `convert` failure.
# Column order counts: a reordering moves the schema downstream operators see.
function checkschema!(st::LastRowState, c::DataFrame)
    cols = names(c)
    if st.names === nothing
        for k in st.keycols
            String(k) in cols ||
                throw(ArgumentError("lastrow key column $k not found in input"))
        end
        st.names = cols
    elseif st.names != cols
        throw(
            ArgumentError("lastrow: chunk columns changed mid-stream, from \
                $(st.names) to $(cols)"),
        )
    end
    return nothing
end

# Function barrier: called with concretely typed arguments, so the per-row work
# compiles to direct column access with nothing boxed. Last write wins, which is
# exactly "each key's last row"; `get!` claims the next slot on a miss, so a row
# costs one hash whether or not its key is new — joinsegment!'s admission loop
# without the tolerance and ordering machinery.
function lastsegment!(index::Dict{K,Int}, slots::Vector{V}, nt::NamedTuple,
    ::Val{KN}) where {K,V,KN}
    for i in eachindex(nt.time)
        row = rowat(V, nt, i)
        j = get!(index, keyat(nt, i, Val(KN)), length(slots) + 1)
        j > length(slots) ? push!(slots, row) : (@inbounds slots[j] = row)
    end
    return nothing
end

function flushlastrow(st::LastRowState, ::Val{KN}, stop) where {KN}
    if isempty(KN)
        st.row === nothing && return nothing
        return DataFrame([retime(st.row, stop)])
    end
    st.index === nothing && return nothing
    ordered = sortedgroups(st.index)
    isempty(ordered) && return nothing
    return DataFrame(emitlast(st.slots, ordered, stop))
end

# Function barrier: the emitted rows are concretely typed, so DataFrame builds
# typed columns from them directly with no promotion pass — and the O(keys) slot
# reads are not O(keys) dynamic dispatches through `st.slots::Any`.
emitlast(slots::Vector{V}, ordered::Vector{<:Pair}, stop) where {V} =
    [retime(@inbounds(slots[j]), stop) for (_, j) in ordered]

# Overwrite the row's time with the emission time. `merge` replaces an existing
# key's value in place, so `:time` keeps the column position it had in the input
# and the output schema is the input's exactly. `stop` already carries the
# context's time type, so nothing is converted.
@inline retime(row::NamedTuple, stop) = merge(row, (; time = stop))
