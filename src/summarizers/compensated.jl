# Floating-point sum accumulators use Kahan-Babuška-Neumaier compensated
# summation over the finite terms only; NaN and ±Inf terms are counted, and
# `compvalue` reconstructs the IEEE result `Base.sum` would give. Keeping
# nonfinites out of the running pair lets `downdate!` remove them exactly
# (folded in, NaN absorbs and an evicted Inf leaves Inf - Inf = NaN). Missing
# terms are counted the same way (the sum state's flag M). BigFloat is
# excluded: compensation buys nothing at arbitrary precision, and a non-isbits
# Compensated{BigFloat} would allocate per row.
compensable(::Type{T}) where {T} = T <: AbstractFloat && T !== BigFloat

# How far a float state's largest folded term may exceed its window's own
# scale before it asks the running tiers to refold it (`needsrefold`; the sum
# family and the co-moment, issue #97): what sliding leaves behind short of a
# refold is then at most about eps·REFOLDRATIO of that scale.
const REFOLDRATIO = 2.0^10

struct Compensated{A<:AbstractFloat}
    total::A     # Neumaier sum of the finite terms only
    comp::A      # running compensation
    nans::Int    # NaN terms currently folded
    posinf::Int  # +Inf terms
    neginf::Int  # -Inf terms
end

compzero(::Type{A}) where {A<:AbstractFloat} =
    Compensated{A}(zero(A), zero(A), 0, 0, 0)

# One Neumaier step. The isfinite guard keeps comp finite when finite terms
# overflow the total to Inf, so compvalue gives Inf, as Base.sum does, not NaN.
@inline function neumaier(s::A, c::A, x::A) where {A<:AbstractFloat}
    t = s + x
    if isfinite(t)
        c += abs(s) >= abs(x) ? (s - t) + x : (x - t) + s
    end
    return t, c
end

@inline function compadd(a::Compensated{A}, x::A) where {A}
    isfinite(x) || return compcount(a, x, 1)
    t, c = neumaier(a.total, a.comp, x)
    return Compensated{A}(t, c, a.nans, a.posinf, a.neginf)
end

# Inverse of compadd over the identically computed term, so the counts always
# balance; negating a finite float is exact.
@inline function compsub(a::Compensated{A}, x::A) where {A}
    isfinite(x) || return compcount(a, x, -1)
    t, c = neumaier(a.total, a.comp, -x)
    return Compensated{A}(t, c, a.nans, a.posinf, a.neginf)
end

# Count the nonfinite term `x` in (`d = 1`) or out (`d = -1`).
@inline function compcount(a::Compensated{A}, x::A, d::Int) where {A}
    isnan(x) && return Compensated{A}(a.total, a.comp, a.nans + d, a.posinf, a.neginf)
    x > zero(A) &&
        return Compensated{A}(a.total, a.comp, a.nans, a.posinf + d, a.neginf)
    return Compensated{A}(a.total, a.comp, a.nans, a.posinf, a.neginf + d)
end

@inline function compmerge(a::Compensated{A}, b::Compensated{A}) where {A}
    t, c = neumaier(a.total, a.comp, b.total)
    t, c = neumaier(t, c, b.comp)
    return Compensated{A}(t, c, a.nans + b.nans, a.posinf + b.posinf,
        a.neginf + b.neginf)
end

@inline function compvalue(a::Compensated{A}) where {A}
    a.nans > 0 && return convert(A, NaN)
    a.posinf > 0 && return a.neginf > 0 ? convert(A, NaN) : convert(A, Inf)
    a.neginf > 0 && return convert(A, -Inf)
    return a.total + a.comp
end

# An accumulator's storage: a `Compensated{A}` for a compensable float, else A
# itself. The operations below dispatch on it.
accstorage(::Type{A}) where {A} = compensable(A) ? Compensated{A} : A
acczero(::Type{S}) where {S} = convert(S, 0)
acczero(::Type{Compensated{A}}) where {A} = compzero(A)
@inline accadd(s, x) = s + x
@inline accadd(s::Compensated, x) = compadd(s, x)
@inline accsub(s, x) = s - x
@inline accsub(s::Compensated, x) = compsub(s, x)
@inline accmerge(a, b) = a + b
@inline accmerge(a::Compensated, b::Compensated) = compmerge(a, b)
@inline accvalue(s) = s
@inline accvalue(s::Compensated) = compvalue(s)

# The windowed sums' storage (`freshwindowed`, for the running tiers): a
# Compensated plus what decides when to refold it (issue #97). The compensation
# is a plain float, so once it holds the round-off of a term far larger than
# the rest it loses whatever lies far below that, and the loss stays after the
# term has gone (pandas GH#52407). So this keeps the largest finite |term|
# folded (`peak`) and a plain Σ|term| over the live finite terms (`mag`, since
# the sum itself may cancel to near 0); the state asks to be refolded once
# peak exceeds REFOLDRATIO times mag. A live term is part of mag, so it asks
# only once a dominant term has left. Only the windowed state carries it: on
# the folds that never downdate it cost up to a quarter of `summarize`'s time.
struct Tracked{A<:AbstractFloat}
    sum::Compensated{A}
    peak::A
    mag::A
end

trackedstorage(::Type{A}) where {A} = compensable(A) ? Tracked{A} : A
acczero(::Type{Tracked{A}}) where {A} = Tracked{A}(compzero(A), zero(A), zero(A))
@inline function accadd(s::Tracked{A}, x::A) where {A}
    f = isfinite(x)
    return Tracked{A}(compadd(s.sum, x), ifelse(f, max(s.peak, abs(x)), s.peak),
        ifelse(f, s.mag + abs(x), s.mag))
end
@inline accsub(s::Tracked{A}, x::A) where {A} =
    Tracked{A}(compsub(s.sum, x), s.peak, ifelse(isfinite(x), s.mag - abs(x), s.mag))
@inline accmerge(a::Tracked{A}, b::Tracked{A}) where {A} =
    Tracked{A}(compmerge(a.sum, b.sum), max(a.peak, b.peak), a.mag + b.mag)
@inline accvalue(s::Tracked) = compvalue(s.sum)
# A mag gone negative or to zero under round-off asks too.
@inline accrefold(s) = false
@inline accrefold(s::Tracked{A}) where {A} = s.peak > convert(A, REFOLDRATIO) * s.mag

# Carry an accumulator into a wider storage. A plain total enters a compensated
# storage as one term; a compensated one keeps its pair and counters.
widenacc(::Type{S2}, acc) where {S2} = convert(S2, accvalue(acc))
widenacc(::Type{Compensated{A2}}, acc) where {A2} =
    compadd(compzero(A2), convert(A2, acc))
widenacc(::Type{Compensated{A2}}, acc::Compensated) where {A2} =
    widencomp(A2, acc)

# A Compensated at a wider float type, with its counters unchanged.
widencomp(::Type{A2}, a::Compensated) where {A2} =
    Compensated{A2}(convert(A2, a.total), convert(A2, a.comp),
        a.nans, a.posinf, a.neginf)

# Error-free products, for the co-moments and AgeWeightedSum. A co-moment
# state's float sums are each `Σd·d` ≈ n·D² for D the distance from the shift
# to the window, while the co-moment is at the scale of the spread s, so any
# rounding at the scale of D² costs eps·(D/s)² (issue #91: a two-row window of
# near-equal values). Hence every product entering the sums is error-free (an
# fma two-product, its error folded into the compensation), and `value` reads
# each Compensated pair `total + comp` as a double-double rather than rounding
# it to one float. Plain storage (integers, BigFloat) uses ordinary arithmetic.
@inline twoprod(x, y) = (p = x * y; (p, fma(x, y, -p)))

# Fold the pair p + e: p takes the Neumaier step, the small e goes straight to
# the compensation. A nonfinite p (overflow) is counted as compadd counts it.
@inline function compaddpair(s::Compensated{A}, p::A, e::A) where {A}
    isfinite(p) || return compadd(s, p)
    t, c = neumaier(s.total, s.comp, p)
    return Compensated{A}(t, c + e, s.nans, s.posinf, s.neginf)
end
# Its inverse over the identically computed pair.
@inline function compsubpair(s::Compensated{A}, p::A, e::A) where {A}
    isfinite(p) || return compsub(s, p)
    t, c = neumaier(s.total, s.comp, -p)
    return Compensated{A}(t, c - e, s.nans, s.posinf, s.neginf)
end

# s ± x·y.
@inline accaddprod(s, x, y) = s + x * y
@inline accaddprod(s::Compensated, x, y) = compaddpair(s, twoprod(x, y)...)
@inline accsubprod(s, x, y) = s - x * y
@inline accsubprod(s::Compensated, x, y) = compsubpair(s, twoprod(x, y)...)

# s - x·y for y an accumulator in s's storage: x times y's total exactly, plus x
# times its compensation, already at eps² of the product.
@inline accsubscaled(s, x, y) = s - x * y
@inline function accsubscaled(s::Compensated{A}, x::A, y::Compensated{A}) where {A}
    p, e = twoprod(x, y.total)
    return compsubpair(s, p, e + x * y.comp)
end

# s + n·x·y, n a row count: n·x is formed exactly first.
@inline accaddprod3(s, n, x, y) = s + n * x * y
@inline function accaddprod3(s::Compensated{A}, n::A, x::A, y::A) where {A}
    q, f = twoprod(n, x)
    p, e = twoprod(q, y)
    return compaddpair(s, p, e + f * y)
end

# Whether no nonfinite term is folded, so the pair is the whole sum.
@inline compfinite(s::Compensated) = s.nans == 0 && s.posinf == 0 && s.neginf == 0

# AgeWeightedSum's S2: its finite pair plus another pair (S1's, or a scaled
# one), as `compmerge` adds, leaving S2's counters at zero.
@inline function pairadd(a::Compensated{A}, total::A, comp::A) where {A}
    t, c = neumaier(a.total, a.comp, total)
    t, c = neumaier(t, c, comp)
    return Compensated{A}(t, c, 0, 0, 0)
end

# S2 - k·x with the product folded error-free (issue #94: a rounded age weight
# loses the same error on every slide of a flat series, so the windowed sum
# drifted linearly with the number of rows slid).
@inline function pairsubprod(a::Compensated{A}, k::A, x::A) where {A}
    p, e = twoprod(k, x)
    return pairadd(a, -p, -e)
end
