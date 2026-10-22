# The centred co-moment `Σ(a - ā)(b - b̄)`: the accumulator behind
# Variance (a = b), Covariance and a LinearRegression with an intercept.
# Unexported, and only ever requested in canonical (`isless`-sorted) order, so
# it needs no alias. The raw-sum identity `Σab - Σa·Σb/n` cancels
# catastrophically when the mean is large against the spread: each `x * x` term
# is already rounded at the scale of x², and Σx² reaches the subtraction
# rounded to one float, so no summation can save it (issue #89;
# notes/variance-cancellation.md). This state folds sums of *shifted* values,
# `d = x - K` for a shift K near the data, over which the same identity is
# exact up to round-off at the scale of the spread.
struct CoMoment{A,B} <: GroupSummarizer end

comomentname(a, b) = Symbol(a, :_, b, :_comoment)
canonicalcomoment(a::Symbol, b::Symbol) =
    ((x, y) = canonicalpair(a, b); CoMoment{x,y}())

emptyvalue(::CoMoment{A,B}) where {A,B} = NamedTuple{(comomentname(A, B),)}((0,))
fresh(s::CoMoment{A,B}, intypes::NamedTuple) where {A,B} =
    comomentfresh(s, dottype(intypes[A], intypes[B]))
freshwindowed(s::CoMoment{A,B}, intypes::NamedTuple) where {A,B} =
    comomentfresh(s, dottype(intypes[A], intypes[B]), Val(true))

# Only rows whose two values are present and finite are folded (`n` of them),
# so every sum runs over the same rows and the shift can be moved as a whole. A
# `missing` row is counted under the flag M, as in `AccumState`, and a
# nonfinite one in `nonfinite` (the value is then NaN, as `Statistics.var`
# gives), so both evict exactly and the state stays a group. The shifts are
# set from the first folded row, and the sums restart at exactly zero whenever
# `n` returns to 0. With A === B, `sb` stays zero and `sa` stands in for it.
#
# A float state recentres once more than half its rows arrived since the shift
# last moved (`age`), moving the shift to the row just folded: a steady window
# of w rows recentres every w/2 rows, a growing fold each time it doubles. It
# is a reparameterization of the same sums, so later `downdate!`s stay
# consistent, and it keeps the shift near the data as the level drifts. The
# new shift is a data value rather than the mean, so deviations stay exact
# differences of inputs: states folding the same rows under different shifts
# then agree exactly where the inputs allow (a flat window is exactly 0).
#
# Sliding can't remove every trace of a row that dominated the window (issue
# #97): its deviation is rounded at its own scale, and removed under a later
# shift it rounds differently, and a compensation term holding its round-off
# loses what is far below it. Either leaves a residue of about eps times the
# largest product folded, which stays once the row has gone. So the windowed
# float state (`freshwindowed`, flag W) keeps that largest product (`peak`,
# since the sums last restarted) and asks
# the running tiers to refold it from the live rows (`needsrefold`) once peak
# exceeds `REFOLDRATIO` times the window's own scale: Σd² for a variance, and
# for a covariance (a ≠ b) √(Σda²·Σdb²), from plain sums kept for this alone
# (`qa`, `qb`), since the co-moment itself may be near 0. While the dominant
# row is live it is part of that scale, so a refold happens only once it has
# left, and the residue left behind otherwise is at most ~eps·REFOLDRATIO of
# the scale. Only the windowed state tracks (W false leaves peak, qa and qb at
# 0): the ordinary one never downdates, and tracking cost `summarize` 8%.
mutable struct CoMomentState{N,A,B,T,M,S,W} <: SummarizerState
    n::Int
    age::Int
    ka::T
    kb::T
    sa::S   # Σ(a - ka)
    sb::S   # Σ(b - kb)
    sab::S  # Σ(a - ka)(b - kb)
    peak::T # the largest |(a - ka)(b - kb)| folded since the sums restarted
    qa::T   # Σ(a - ka)², plain; kept for A !== B only
    qb::T   # Σ(b - kb)², plain; kept for A !== B only (all three: W only)
    nonfinite::Int
    missings::Int
end

# The state for accumulator type T, chosen as `accumfresh` chooses; the
# windowed one (`Val(true)`) tracks what decides a refold, for a float.
function comomentfresh(s::CoMoment{A,B}, ::Type{T},
    ::Val{windowed} = Val(false)) where {A,B,T,windowed}
    N = comomentname(A, B)
    T === Missing && return MissingState{N}(s)
    M = Missing <: T
    Tn = nonmissingtype(T)
    return CoMomentState{N,A,B,Tn,M,accstorage(Tn),windowed && compensable(Tn)}()
end

CoMomentState{N,A,B,T,M,S,W}() where {N,A,B,T,M,S,W} =
    CoMomentState{N,A,B,T,M,S,W}(0, 0, convert(T, 0), convert(T, 0), acczero(S),
        acczero(S), acczero(S), convert(T, 0), convert(T, 0), convert(T, 0), 0, 0)

# The co-moment from the shifted sums, with one rounding: the numerator is
# exact for integers (and for floats whose products are), so states holding
# the same rows under different shifts agree exactly on a zero co-moment.
@inline comomentof(n::Int, sab, sa, sb) = (n * sab - sa * sb) / n
# The value's type, from that arithmetic: integer input gives Float64.
comomenttype(::Type{T}) where {T} = Base.promote_op(comomentof, Int, T, T, T)


# `comomentof` over the sums as stored: (n·sab - sa·sb)/n in double-double for
# compensated sums, the leading products formed exactly. With exact sums and
# products every low part is 0 and this is `comomentof` itself, so the exact
# agreement on a zero co-moment survives. A sum holding an overflowed term falls
# back to the IEEE reconstruction.
@inline comomentvalue(n::Int, sab, sa, sb) = comomentof(n, sab, sa, sb)
@inline function comomentvalue(n::Int, sab::Compensated{A}, sa::Compensated{A},
    sb::Compensated{A}) where {A}
    compfinite(sab) && compfinite(sa) && compfinite(sb) ||
        return comomentof(n, compvalue(sab), compvalue(sa), compvalue(sb))
    nf = convert(A, n)
    p1, e1 = twoprod(nf, sab.total)
    p2, e2 = twoprod(sa.total, sb.total)
    lo = (e1 + nf * sab.comp) - (e2 + (sa.total * sb.comp + sa.comp * sb.total))
    return ((p1 - p2) + lo) / n
end


fresh(::S) where {S<:CoMomentState} = S()
@inline function fresh!(st::CoMomentState{N,A,B,T,M,S}) where {N,A,B,T,M,S}
    st.n = 0
    st.age = 0
    st.sa = acczero(S)
    st.sb = acczero(S)
    st.sab = acczero(S)
    st.peak = st.qa = st.qb = zero(T)
    st.nonfinite = 0
    st.missings = 0
    return st
end

@inline function update!(st::CoMomentState{N,A,B,T,M,S,W},
    row) where {N,A,B,T,M,S,W}
    va = getproperty(row, A)
    vb = getproperty(row, B)
    if M && (ismissing(va) || ismissing(vb))
        st.missings += 1
        return nothing
    end
    a = convert(T, va)
    b = convert(T, vb)
    if !(isfinite(a) && isfinite(b))
        st.nonfinite += 1
        return nothing
    end
    if st.n == 0
        st.ka = a
        st.kb = b
    else
        st.age += 1
    end
    da = a - st.ka
    db = b - st.kb
    st.sa = accadd(st.sa, da)
    A === B || (st.sb = accadd(st.sb, db))
    st.sab = accaddprod(st.sab, da, db)
    if W
        st.peak = max(st.peak, abs(da * db))
        A === B || (st.qa += da * da; st.qb += db * db)
    end
    st.n += 1
    compensable(T) && 2 * st.age > st.n && recentre!(st, a, b)
    return nothing
end

@inline function downdate!(st::CoMomentState{N,A,B,T,M,S,W},
    row) where {N,A,B,T,M,S,W}
    va = getproperty(row, A)
    vb = getproperty(row, B)
    if M && (ismissing(va) || ismissing(vb))
        st.missings -= 1
        return nothing
    end
    a = convert(T, va)
    b = convert(T, vb)
    if !(isfinite(a) && isfinite(b))
        st.nonfinite -= 1
        return nothing
    end
    st.n -= 1
    if st.n == 0
        st.age = 0
        st.sa = acczero(S)
        st.sb = acczero(S)
        st.sab = acczero(S)
        st.peak = st.qa = st.qb = zero(T)
    else
        da = a - st.ka
        db = b - st.kb
        st.sa = accsub(st.sa, da)
        A === B || (st.sb = accsub(st.sb, db))
        st.sab = accsubprod(st.sab, da, db)
        W && A !== B && (st.qa -= da * da; st.qb -= db * db)
    end
    return nothing
end

# Peak against the window's scale, as the comment on the struct describes,
# squared for a covariance. A scale gone negative or to zero under round-off
# asks too; a NaN one (an overflowed sum) doesn't, since refolding would not
# change it. Read on every eviction, so the compensated pair is read as is.
@inline function needsrefold(st::CoMomentState{N,A,B,T,M,S,W}) where {N,A,B,T,M,S,W}
    W || return false
    r = convert(T, REFOLDRATIO)
    A === B && return st.peak > r * (st.sab.total + st.sab.comp)
    return st.peak * st.peak > r * r * max(st.qa * st.qb, zero(T))
end

# Move the shifts to (ka, kb). With δ the move,
# Σ(a - ka - δa)(b - kb - δb) = Σab - δb·Σa - δa·Σb + n·δa·δb, each term folded
# into the compensated sums exactly (to eps² of the term).
@inline function recentre!(st::CoMomentState{N,A,B,T,M,S,W}, ka::T,
    kb::T) where {N,A,B,T,M,S,W}
    n = convert(T, st.n)
    sa = st.sa
    sb = A === B ? sa : st.sb
    δa = ka - st.ka
    δb = kb - st.kb
    # Σ(d - δ)² = Σd² - 2δ·Σd + n·δ², from the sums before the move.
    if W && A !== B
        st.qa += δa * (n * δa - 2 * accvalue(sa))
        st.qb += δb * (n * δb - 2 * accvalue(sb))
    end
    st.sab = accaddprod3(accsubscaled(accsubscaled(st.sab, δb, sa), δa, sb),
        n, δa, δb)
    st.sa = accsubprod(st.sa, n, δa)
    A === B || (st.sb = accsubprod(st.sb, n, δb))
    st.ka = ka
    st.kb = kb
    st.age = 0
    return nothing
end

# Reads every input before writing, since `dest` may alias either. b's sums
# are re-expressed at a's shifts (δ = b's shift - a's) by the same expansion as
# `recentre!`; an empty side contributes only its counts.
function combine!(dest::CoMomentState{N,A,B,T,M,S,W},
    a::CoMomentState{N,A,B,T,M,S,W},
    b::CoMomentState{N,A,B,T,M,S,W}) where {N,A,B,T,M,S,W}
    nonfinite = a.nonfinite + b.nonfinite
    missings = a.missings + b.missings
    if b.n == 0
        n, age, ka, kb, sa, sb, sab = a.n, a.age, a.ka, a.kb, a.sa, a.sb, a.sab
        peak, qa, qb = a.peak, a.qa, a.qb
    elseif a.n == 0
        n, age, ka, kb, sa, sb, sab = b.n, b.age, b.ka, b.kb, b.sa, b.sb, b.sab
        peak, qa, qb = b.peak, b.qa, b.qb
    else
        nb = convert(T, b.n)
        δa = b.ka - a.ka
        δb = b.kb - a.kb
        bsb = A === B ? b.sa : b.sb
        sa = accaddprod(accmerge(a.sa, b.sa), nb, δa)
        sb = A === B ? a.sb : accaddprod(accmerge(a.sb, b.sb), nb, δb)
        # + δa·Σb + δb·Σa, as the subtraction of the negated shift (exact).
        sab = accaddprod3(
            accsubscaled(accsubscaled(accmerge(a.sab, b.sab), -δa, bsb), -δb, b.sa),
            nb, δa, δb)
        n, age, ka, kb = a.n + b.n, a.age + b.n, a.ka, a.kb
        # b's plain squares at a's shifts: Σ(d + δ)² = Σd² + δ·(2Σd + n·δ).
        peak = max(a.peak, b.peak)
        qa = qb = zero(T)
        if W && A !== B
            qa = a.qa + (b.qa + δa * (2 * accvalue(b.sa) + nb * δa))
            qb = a.qb + (b.qb + δb * (2 * accvalue(b.sb) + nb * δb))
        end
    end
    dest.n, dest.age, dest.ka, dest.kb = n, age, ka, kb
    dest.sa, dest.sb, dest.sab = sa, sb, sab
    dest.peak, dest.qa, dest.qb = peak, qa, qb
    dest.nonfinite, dest.missings = nonfinite, missings
    return nothing
end

# `missing` while a missing row is folded, else NaN while a nonfinite one is.
# A variance (A === B) is clamped at zero, so round-off never makes it
# negative.
@inline function value(st::CoMomentState{N,A,B,T,M}) where {N,A,B,T,M}
    V = comomenttype(T)
    R = maybemissing(V, M)
    M && st.missings > 0 && return NamedTuple{(N,),Tuple{R}}((missing,))
    st.nonfinite > 0 && return NamedTuple{(N,),Tuple{R}}((convert(V, NaN),))
    # One row deviates from its own mean by exactly 0, which sliding sums
    # need not reproduce bit for bit (Σd² minus d·d leaves round-off), and a
    # corrected single-row variance must be 0/0 = NaN, not ±Inf.
    st.n <= 1 && return NamedTuple{(N,),Tuple{R}}((zero(V),))
    c = comomentvalue(st.n, st.sab, st.sa, A === B ? st.sa : st.sb)
    return NamedTuple{(N,),Tuple{R}}((A === B ? max(c, zero(c)) : c,))
end

function widenstate(st::CoMomentState{N,A,B}, intypes::NamedTuple) where {N,A,B}
    w = comomentfresh(CoMoment{A,B}(), dottype(intypes[A], intypes[B]))
    typeof(w) === typeof(st) && return st
    return comomentwiden!(w, st)
end
function comomentwiden!(
    w::CoMomentState{N,A,B,T,M,S},
    st::CoMomentState,
) where {N,A,B,T,M,S}
    w.n, w.age = st.n, st.age
    w.ka, w.kb = convert(T, st.ka), convert(T, st.kb)
    w.sa = widenacc(S, st.sa)
    w.sb = widenacc(S, st.sb)
    w.sab = widenacc(S, st.sab)
    w.peak, w.qa, w.qb = convert(T, st.peak), convert(T, st.qa), convert(T, st.qb)
    w.nonfinite, w.missings = st.nonfinite, st.missings
    return w
end

withmissings!(w::CoMomentState, n::Int) = (w.missings = n; w)
