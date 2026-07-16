"""
    CausalFrame(ctx::Context, chunks)
    CausalFrame(ctx::Context, df::DataFrame)

A materialized time-series table over the window `ctx`. Backed by one or
more time-disjoint DataFrame chunks; the chunk structure is an
implementation detail and is not part of the public API.

Invariants, checked at construction:

- every chunk has a `:time` column with element type `<: T`;
- all (non-empty) chunks share the same column names (element types may
  differ between chunks; `DataFrame(frame)` promotes on concatenation);
- time is non-decreasing within each chunk and across chunk boundaries;
- all times lie in the closed interval `[ctx.start, ctx.stop]`.

Access the data via the Tables.jl interface, `DataFrame(frame)`,
[`context`](@ref), `nrow(frame)`, or `names(frame)`. The constructor takes
ownership of the passed DataFrames; do not mutate them afterwards.
"""
struct CausalFrame{T}
    context::Context{T}
    chunks::Vector{DataFrame}

    function CausalFrame{T}(ctx::Context{T}, chunks::Vector{DataFrame}) where {T}
        kept = DataFrame[c for c in chunks if nrow(c) > 0]
        for c in kept
            "time" in names(c) ||
                throw(ArgumentError("every chunk must have a :time column"))
            eltype(c.time) <: T || throw(ArgumentError(
                "chunk :time column has element type $(eltype(c.time)), expected <: $T"))
            issorted(c.time) ||
                throw(ArgumentError("chunk :time column is not non-decreasing"))
            names(c) == names(first(kept)) ||
                throw(ArgumentError("all chunks must share the same schema"))
            first(c.time) >= ctx.start && last(c.time) <= ctx.stop ||
                throw(ArgumentError(
                    "chunk times must lie in [$(ctx.start), $(ctx.stop)]"))
        end
        for i in 2:length(kept)
            last(kept[i - 1].time) <= first(kept[i].time) ||
                throw(ArgumentError("chunks must be non-decreasing across boundaries"))
        end
        return new{T}(ctx, kept)
    end
end

CausalFrame(ctx::Context{T}, chunks::Vector{DataFrame}) where {T} =
    CausalFrame{T}(ctx, chunks)
CausalFrame(ctx::Context, df::DataFrame) = CausalFrame(ctx, [df])

"""
    context(frame::CausalFrame) -> Context

The time window this frame was loaded over.
"""
context(frame::CausalFrame) = frame.context

timetype(::CausalFrame{T}) where {T} = T

DataFrames.nrow(frame::CausalFrame) = sum(nrow, frame.chunks; init = 0)

Base.names(frame::CausalFrame) =
    isempty(frame.chunks) ? ["time"] : names(first(frame.chunks))

"""
    DataFrame(frame::CausalFrame)

Concatenate the frame's chunks into a plain `DataFrame` — an explicit exit
from the causal world, and the point where the data is copied. A frame with
no rows yields a zero-row DataFrame with only its `:time` column.
"""
function DataFrames.DataFrame(frame::CausalFrame{T}) where {T}
    isempty(frame.chunks) && return DataFrame(time = T[])
    length(frame.chunks) == 1 && return copy(only(frame.chunks))
    return reduce(vcat, frame.chunks)
end

Tables.istable(::Type{<:CausalFrame}) = true
Tables.rowaccess(::Type{<:CausalFrame}) = true
Tables.rows(frame::CausalFrame) = Tables.rows(DataFrame(frame))
Tables.columnaccess(::Type{<:CausalFrame}) = true
Tables.columns(frame::CausalFrame) = Tables.columns(DataFrame(frame))

function Base.show(io::IO, mime::MIME"text/plain", frame::CausalFrame{T}) where {T}
    ctx = frame.context
    println(io, "CausalFrame{$T} with $(nrow(frame)) rows over [$(ctx.start), $(ctx.stop)]")
    return show(io, mime, DataFrame(frame); summary = false)
end
