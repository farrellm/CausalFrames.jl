"""
    CausalFrame(ctx::Context, df::DataFrame)

A materialized time-series table over the window `ctx`, backed by a single
DataFrame. **Opaque**: the backing storage is an implementation detail and is
not part of the public API.

Invariants, checked at construction:

- the DataFrame has a `:time` column with element type `<: T`;
- time is non-decreasing;
- all times lie in the closed interval `[ctx.start, ctx.stop]`.

Access the data via the Tables.jl interface, `DataFrame(frame)`,
[`context`](@ref), `nrow(frame)`, or `names(frame)`. The constructor takes
ownership of the passed DataFrame; do not mutate it afterwards.
"""
struct CausalFrame{T}
    context::Context{T}
    data::DataFrame

    function CausalFrame{T}(ctx::Context{T}, df::DataFrame) where {T}
        "time" in names(df) ||
            throw(ArgumentError("frame must have a :time column"))
        eltype(df.time) <: T || throw(ArgumentError(
            ":time column has element type $(eltype(df.time)), expected <: $T"))
        issorted(df.time) ||
            throw(ArgumentError(":time column is not non-decreasing"))
        nrow(df) == 0 || (first(df.time) >= ctx.start && last(df.time) <= ctx.stop) ||
            throw(ArgumentError(
                "times must lie in [$(ctx.start), $(ctx.stop)]"))
        return new{T}(ctx, df)
    end
end

CausalFrame(ctx::Context{T}, df::DataFrame) where {T} = CausalFrame{T}(ctx, df)

"""
    context(frame::CausalFrame) -> Context

The time window this frame was loaded over.
"""
context(frame::CausalFrame) = frame.context

timetype(::CausalFrame{T}) where {T} = T

DataFrames.nrow(frame::CausalFrame) = nrow(frame.data)

Base.names(frame::CausalFrame) = names(frame.data)

"""
    DataFrame(frame::CausalFrame)

Copy the frame's data into a plain `DataFrame` — an explicit exit from the
causal world.
"""
DataFrames.DataFrame(frame::CausalFrame) = copy(frame.data)

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
