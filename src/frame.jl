# Internal token: the caller vouches for the chunk-protocol invariants
# (non-empty chunks, shared schema, ordered in-window times), so the trusted
# constructor may skip the O(n) validation. Only load and stream qualify —
# they consume iterators that already guarantee the protocol.
struct Trusted end

"""
    CausalFrame(ctx::Context, chunks::Vector{DataFrame})
    CausalFrame(ctx::Context, df::DataFrame)

A materialized time-series table over the window `ctx`, usually obtained from
[`load`](@ref) rather than constructed directly.

# Arguments
- `ctx`: the window the data covers.
- `chunks` / `df`: the data, as time-ordered DataFrames. The frame takes
  ownership of them; do not mutate them afterwards.

The constructor checks, throwing an `ArgumentError` otherwise, that every
non-empty chunk has a `:time` column with element type `<: T`, that all
non-empty chunks share their column names (element types may differ;
`DataFrame(frame)` promotes them), that time is non-decreasing within and
across chunks, and that every time lies in `[ctx.start, ctx.stop]`.

The backing chunks are not part of the API. Read the data through the Tables.jl
interface or `DataFrame(frame)`, and query it with [`context`](@ref),
`nrow(frame)` and `names(frame)`.
"""
struct CausalFrame{T}
    context::Context{T}
    chunks::Vector{DataFrame}

    function CausalFrame{T}(ctx::Context{T}, chunks::Vector{DataFrame}) where {T}
        kept = DataFrame[c for c in chunks if nrow(c) > 0]
        schema = isempty(kept) ? String[] : names(first(kept))
        for c in kept
            "time" in names(c) ||
                throw(ArgumentError("every chunk must have a :time column"))
            eltype(c.time) <: T || throw(
                ArgumentError(
                    "chunk :time column has element type $(eltype(c.time)), expected <: $T",
                ),
            )
            issorted(c.time) ||
                throw(ArgumentError("chunk :time column is not non-decreasing"))
            names(c) == schema ||
                throw(ArgumentError("all chunks must share the same schema"))
            first(c.time) >= ctx.start && last(c.time) <= ctx.stop ||
                throw(ArgumentError(
                    "chunk times must lie in [$(ctx.start), $(ctx.stop)]"))
        end
        for i in 2:length(kept)
            last(kept[i-1].time) <= first(kept[i].time) ||
                throw(ArgumentError("chunks must be non-decreasing across boundaries"))
        end
        return new{T}(ctx, kept)
    end

    CausalFrame{T}(::Trusted, ctx::Context{T}, chunks::Vector{DataFrame}) where {T} =
        new{T}(ctx, chunks)
end

CausalFrame(ctx::Context{T}, chunks::Vector{DataFrame}) where {T} =
    CausalFrame{T}(ctx, chunks)
CausalFrame(ctx::Context, df::DataFrame) = CausalFrame(ctx, [df])

"""
    context(frame::CausalFrame) -> Context

The window the frame was loaded over.
"""
context(frame::CausalFrame) = frame.context

timetype(::CausalFrame{T}) where {T} = T

DataFrames.nrow(frame::CausalFrame) = sum(nrow, frame.chunks; init = 0)

Base.names(frame::CausalFrame) =
    isempty(frame.chunks) ? ["time"] : names(first(frame.chunks))

"""
    DataFrame(frame::CausalFrame) -> DataFrame

Copy the frame into a plain `DataFrame`, promoting column element types across
chunks. A frame with no rows gives a zero-row DataFrame with only `:time`.
"""
function DataFrames.DataFrame(frame::CausalFrame{T}) where {T}
    isempty(frame.chunks) && return DataFrame(time = T[])
    length(frame.chunks) == 1 && return copy(only(frame.chunks))
    return reduce(vcat, frame.chunks)
end

# Column access only: rows are served through Tables.jl's row-view fallback
# over the columns, so consumers touching both pay one materialization, not
# two. The copy in Tables.columns keeps the backing opaque.
Tables.istable(::Type{<:CausalFrame}) = true
Tables.columnaccess(::Type{<:CausalFrame}) = true
Tables.columns(frame::CausalFrame) = Tables.columns(DataFrame(frame))

# The names are known without materializing, and the eltypes are the
# promotion of each column's per-chunk eltypes — what `DataFrame(frame)`
# produces on concatenation — so the schema costs O(chunks * columns), never
# a row scan.
function Tables.schema(frame::CausalFrame{T}) where {T}
    isempty(frame.chunks) && return Tables.Schema((:time,), (T,))
    ns = propertynames(first(frame.chunks))
    types = [mapreduce(c -> eltype(c[!, n]), promote_type, frame.chunks)
             for n in ns]
    return Tables.Schema(ns, types)
end

# One partition per backing chunk; the copies keep the backing opaque. An
# empty frame yields the single zero-row frame `DataFrame(frame)` would, so
# partition-aware sinks see the same table as whole-table consumers.
Tables.partitions(frame::CausalFrame{T}) where {T} =
    (copy(c) for c in (isempty(frame.chunks) ? (DataFrame(time = T[]),) :
                       frame.chunks))

function Base.show(io::IO, mime::MIME"text/plain", frame::CausalFrame{T}) where {T}
    ctx = frame.context
    println(io, "CausalFrame{$T} with $(nrow(frame)) rows over [$(ctx.start), $(ctx.stop)]")
    return show(io, mime, DataFrame(frame); summary = false)
end
