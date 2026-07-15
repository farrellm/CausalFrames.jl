"""
    Context(start, stop)

A time window over which a [`CausalPipeline`](@ref) is evaluated. The time
type `T` may be any ordered type (`DateTime`, `Int`, `Float64`, ...).
`start <= stop` is enforced.

Sources clip their output to the half-open interval `[start, stop)`; a
materialized [`CausalFrame`](@ref) may contain rows in the closed interval
`[start, stop]` (intermediate operators may emit a row exactly at `stop`).
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
