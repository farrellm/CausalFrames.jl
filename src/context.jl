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

# The context an operator reads its input over when it must see `amount` before
# `start`: the rows a `tolerance` may still match, or a look-back's worth.
# Non-negativity is probed by subtracting rather than by comparing with zero,
# so an amount with no zero of its own (a `Period`, say) works; `what` names
# the operator and argument in the error. `nothing` widens nothing.
widenstart(ctx::Context, ::Nothing, what::String) = ctx
function widenstart(ctx::Context, amount, what::String)
    start = ctx.start - amount
    start <= ctx.start ||
        throw(ArgumentError("$what must be non-negative, got $amount"))
    return Context(start, ctx.stop)
end
