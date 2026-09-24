"""
    Context(start, stop)

The time window a [`CausalPipeline`](@ref) is evaluated over.

# Arguments
- `start`, `stop`: the window bounds, promoted to a common time type `T`. Any
  ordered type works (`DateTime`, `Int`, `Float64`, …); `start <= stop` is
  required.

Sources clip their output to the half-open interval `[start, stop)`. A loaded
[`CausalFrame`](@ref) may also hold rows at `stop`, since some transforms (such
as [`summarize`](@ref)) emit there.
"""
struct Context{T}
    start::T
    stop::T
    function Context{T}(start, stop) where {T}
        start <= stop ||
            throw(ArgumentError("Context start ($start) must be <= stop ($stop)"))
        return new{T}(start, stop)
    end
end

Context(start::T, stop::T) where {T} = Context{T}(start, stop)
Context(start, stop) = Context(promote(start, stop)...)

"""
    timetype(ctx::Context{T}) -> T

The time type of a context.
"""
timetype(::Context{T}) where {T} = T
