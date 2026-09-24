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

# Calendar periods have no fixed length: `Month(1)` is 28 to 31 days. Dates
# can add one to a time but not compare one with a time difference (a `Day` or
# `Millisecond`), so the windows below measure them on the calendar instead.
const CalendarPeriod = Dates.OtherPeriod   # Month, Quarter, Year

# Whether `s` lies at most `lb` before `t`: the membership test of every
# backward window (asofjoin and forwardfill tolerance, rolling and window
# look-backs). The fixed form is kept verbatim rather than rearranged, since
# `s >= t - lb` can disagree at the last ulp for floating-point times; the
# calendar form is exact and uses the same arithmetic as the widened context
# (`start - lb`). Both are monotone in `s` and in `t`, which the eviction heads
# and the segment tree's binary search rely on.
@inline withinback(t, s, lb) = t - s <= lb
@inline withinback(t, s, lb::CalendarPeriod) = s >= t - lb

# Whether `r` lies at most `lb` after `t`: futurejoin's forward tolerance,
# matching its widened context (`stop + lb`) for calendar periods.
@inline withinahead(t, r, lb) = r - t <= lb
@inline withinahead(t, r, lb::CalendarPeriod) = r <= t + lb
