# The last-row-per-key transform. Like `summarize` it folds the whole window and
# emits once the stream is exhausted, but it keeps rows rather than summaries,
# so its store is join.jl's `SlotStore` (see there, and DESIGN.md, "Representing
# a match"), not a GroupTable. A slot is written as soon as its key claims it,
# so unlike the join's matches no `found` mask is needed.

"""
    lastrow(; key = nothing) -> (CausalPipeline -> CausalPipeline)
    lastrow(p::CausalPipeline; key = nothing) -> CausalPipeline

A transform emitting the last row of the window, retimed to `stop`, with every
column kept. An empty input emits nothing. It emits once its input is
exhausted, so it streams as a single frame.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, emit each key's last row, sorted by key.

The original time is overwritten; keep it under another name if needed:

```jldoctest
df = DataFrame(time = [1, 2, 3], sym = ["a", "b", "a"], px = [10, 20, 11])
p = readtable(df) |> addcolumns(r -> (; t0 = r.time)) |> lastrow(; key = :sym)
DataFrame(load(Context(0, 10), p))

# output

2×4 DataFrame
 Row │ time   sym     px     t0
     │ Int64  String  Int64  Int64
─────┼─────────────────────────────
   1 │    10  a          11      3
   2 │    10  b          20      2
```

A chunk whose column names differ from the first chunk's is an
`ArgumentError`.
"""
function lastrow(; key = nothing)
    keycols = keycolumns(key, "lastrow")
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

# Per-run state. The store's row and key types come from the promoted input
# schema, known only once a chunk arrives, so these fields are untyped per-chunk
# setup; the per-row work sits behind the `lastsegment!` function barrier.
mutable struct LastRowState
    const keycols::Vector{Symbol}
    names::Union{Nothing,Vector{String}}  # column names, fixed by the first chunk
    types::Union{Nothing,NamedTuple}      # promotion of every input schema seen
    store::Any    # SlotStore{K,V}: per-key slot holding that key's last row
    row::Any      # keyless: the last row seen, as a V
end
LastRowState(keycols::Vector{Symbol}) =
    LastRowState(keycols, nothing, nothing, nothing, nothing)

function lastrowchunk!(st::LastRowState, ::Val{KN}, c::DataFrame) where {KN}
    checkschema!(st, c)
    types = promotetypes(st.types, chunktypes(c))
    widened = st.types !== nothing && types != st.types
    st.types = types
    V = storerowtype(types)
    nt = Tables.columntable(c)
    if isempty(KN)
        # Keyless: only the chunk's last row matters. O(ncols) per chunk.
        st.row = rowat(V, nt, length(nt.time))
        return nothing
    end
    if st.store === nothing
        st.store = SlotStore{storekeytype(types, Val(KN)),V}()
    elseif widened
        st.store = widenstore(st.store, storekeytype(types, Val(KN)), V)
    end
    lastsegment!(st.store.index, st.store.slots, nt, Val(KN))
    return nothing
end

# O(ncols) per chunk. Keys are checked on the first chunk; names, including
# their order, on every chunk, since they fix the store's row type and a change
# would otherwise surface as an opaque `convert` failure.
function checkschema!(st::LastRowState, c::DataFrame)
    cols = names(c)
    if st.names === nothing
        checkkeycolumns(st.keycols, c, "lastrow")
        st.names = cols
    elseif st.names != cols
        throw(
            ArgumentError("lastrow: chunk columns changed mid-stream, from \
                $(st.names) to $(cols)"),
        )
    end
    return nothing
end

# Function barrier over concretely typed arguments. Last write wins: the
# join's admission loop without tolerance or ordering.
function lastsegment!(index::Dict{K,Int}, slots::Vector{V}, nt::NamedTuple,
    ::Val{KN}) where {K,V,KN}
    for i in eachindex(nt.time)
        admitslot!(index, slots, keyat(nt, i, Val(KN)), rowat(V, nt, i))
    end
    return nothing
end

function flushlastrow(st::LastRowState, ::Val{KN}, stop) where {KN}
    if isempty(KN)
        st.row === nothing && return nothing
        return DataFrame([retime(st.row, stop)])
    end
    st.store === nothing && return nothing
    ordered = sortedgroups(st.store.index)
    isempty(ordered) && return nothing
    return DataFrame(emitlast(st.store.slots, ordered, stop))
end

# Function barrier: the rows are concretely typed, so DataFrame builds typed
# columns directly and the slot reads don't dispatch through `st.store::Any`.
emitlast(slots::Vector{V}, ordered::Vector{<:Pair}, stop) where {V} =
    [retime(@inbounds(slots[j]), stop) for (_, j) in ordered]

# Set the row's time to the emission time. `merge` keeps `:time` in its input
# position, so the output schema is the input's.
@inline retime(row::NamedTuple, stop) = merge(row, (; time = stop))
