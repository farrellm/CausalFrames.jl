# The centred power sums `Σ(x - x̄)^k` for k = 2, 3, 4, and the shape
# summarizers Skewness and Kurtosis over them.
#
# `CentralMoments{c}` is the one-column, fourth-order sibling of `CoMoment`
# (comoment.jl, whose comments explain the design at length): it folds sums of
# *shifted* values `d = x - K`, for a shift K near the data, so the raw-moment
# expansion doesn't cancel under a large level (issue #89). Unexported, like
# `CoMoment`; its value is the triple `(m2, m3, m4)` of centred sums, read by
# the dependents.
struct CentralMoments{C} <: GroupSummarizer end

centralname(c) = Symbol(c, :_centralmoments)

emptyvalue(::CentralMoments{C}) where {C} = NamedTuple{(centralname(C),)}((missing,))
fresh(s::CentralMoments{C}, intypes::NamedTuple) where {C} =
    centralfresh(s, intypes[C])
freshwindowed(s::CentralMoments{C}, intypes::NamedTuple) where {C} =
    centralfresh(s, intypes[C], Val(true))

# As in `CoMomentState`: only present, finite rows are folded (`n` of them),
# `missing` rows are counted under the flag M and nonfinite ones in
# `nonfinite`, the shift is set from the first folded row, the sums restart at
# exactly zero whenever `n` returns to 0, and a float state recentres to the
# row just folded once more than half its rows postdate the shift.
#
# The fold is at `float(T)`: a fourth power overflows Int64 once a deviation
# passes 55,000 (550 dollars in cents), and the value is a float anyway. A
# float fold rounds nothing at the scale of the distance D from the shift to
# the window, where the sums sit, against moments at the spread's: the
# deviation is an exact pair (`deviation`), each power enters the sums to about
# eps² (`powerpairs`), and recentring and `combine!` move the shift by an exact
# δ with double-double coefficients (`shiftpowers`).
#
# So a departed row leaves only about n·eps²·d⁴ behind, not the eps·d² a
# `CoMoment` row leaves, and the windowed float state (flag W) asks to be
# refolded from its live rows (issue #97) only once its largest folded d⁴
# (`peak`, since the sums last restarted) exceeds `REFOLDRATIO²` times the live
# `Σd⁴`: the co-moment's ratio on |d|, so it asks about as often (on 10⁶ rows
# of a random walk, 351 times at `Bars(5)` against the variance's 213, never at
# `Bars(20)`; on Cauchy steps 1.1x-1.6x as often), where comparing with
# `REFOLDRATIO` itself asked up to 160 times as often. The residue short of the
# ratio is then about n·eps²·REFOLDRATIO² ≈ n·5e-26 of `Σd⁴`. It also refolds
# a window gone flat after a spread, so a flat window's sums are exactly 0
# again and Kurtosis reads it as flat.
mutable struct CentralMomentState{N,C,T,M,S,W} <: SummarizerState
    n::Int
    age::Int
    k::T
    s1::S   # Σd
    s2::S   # Σd²
    s3::S   # Σd³
    s4::S   # Σd⁴
    peak::T # the largest finite d⁴ folded since the sums restarted (W only)
    nonfinite::Int
    missings::Int
end

# `Base.promote_op` rather than `float`, which has no method for `Any`.
centralfloat(::Type{T}) where {T} = Base.promote_op(float, T)

function centralfresh(s::CentralMoments{C}, ::Type{T},
    ::Val{windowed} = Val(false)) where {C,T,windowed}
    N = centralname(C)
    T === Missing && return MissingState{N}(s)
    M = Missing <: T
    Tn = centralfloat(nonmissingtype(T))
    return CentralMomentState{N,C,Tn,M,accstorage(Tn),windowed && compensable(Tn)}()
end

CentralMomentState{N,C,T,M,S,W}() where {N,C,T,M,S,W} =
    CentralMomentState{N,C,T,M,S,W}(0, 0, convert(T, 0), acczero(S), acczero(S),
        acczero(S), acczero(S), convert(T, 0), 0, 0)

fresh(::S) where {S<:CentralMomentState} = S()
@inline function fresh!(st::CentralMomentState{N,C,T,M,S}) where {N,C,T,M,S}
    st.n = 0
    st.age = 0
    st.s1 = st.s2 = st.s3 = st.s4 = acczero(S)
    st.peak = zero(T)
    st.nonfinite = 0
    st.missings = 0
    return st
end

# The deviation d = x - k, and the shift's move δ = k - kold, exactly: as a
# pair (hi, lo) for compensated storage. A plain d would be rounded at its own
# scale D, and a row folded under one shift and evicted under another would
# round differently, leaving eps·D⁴ in Σd⁴ each time: once x crosses zero, or
# spans more than a factor of 2 from the shift, Sterbenz no longer makes the
# subtraction exact, and the residue random-walked (6e-12 of a four-row
# kurtosis over 400 rows of a series swinging through zero).
@inline deviation(::Type{S}, x, k) where {S} = x - k
@inline deviation(::Type{<:Compensated}, x, k) = twosum(x, -k)

# d², d³ and d⁴ of d = dh + dl as pairs `p + e`, each to about eps² of the
# power.
@inline function powerpairs(dh, dl)
    p2, f2 = twoprod(dh, dh)
    e2 = f2 + 2 * dh * dl
    p3, f3 = twoprod(p2, dh)
    p4, f4 = twoprod(p2, p2)
    return p2, e2, p3, f3 + (e2 * dh + p2 * dl), p4, f4 + 2 * p2 * e2
end

# Fold the powers of d into (`sign = 1`) or out of (`-1`) the four sums. Plain
# storage (BigFloat, an abstract type) uses ordinary arithmetic.
@inline function foldpowers(s1, s2, s3, s4, d, sign::Int)
    d2 = d * d
    return (s1 + sign * d, s2 + sign * d2, s3 + sign * (d2 * d),
        s4 + sign * (d2 * d2))
end
@inline function foldpowers(s1::Compensated{A}, s2::Compensated{A},
    s3::Compensated{A}, s4::Compensated{A}, (dh, dl)::Tuple{A,A},
    sign::Int) where {A}
    p2, e2, p3, e3, p4, e4 = powerpairs(dh, dl)
    sign > 0 && return compaddpair(s1, dh, dl), compaddpair(s2, p2, e2),
    compaddpair(s3, p3, e3), compaddpair(s4, p4, e4)
    return compsubpair(s1, dh, dl), compsubpair(s2, p2, e2),
    compsubpair(s3, p3, e3), compsubpair(s4, p4, e4)
end

# d⁴, for the refold ask.
@inline fourth(d) = (q = d * d; q * q)
@inline fourth((dh, dl)::Tuple) = fourth(dh)

# The sums over n rows re-expressed at a shift moved by δ (so each d becomes
# d - δ), by the binomial expansion, from the sums before the move:
#   Σ(d-δ)   = S1 - nδ
#   Σ(d-δ)²  = S2 - 2δ·S1 + nδ²
#   Σ(d-δ)³  = S3 - 3δ·S2 + 3δ²·S1 - nδ³
#   Σ(d-δ)⁴  = S4 - 4δ·S3 + 6δ²·S2 - 4δ³·S1 + nδ⁴
@inline function shiftpowers(n::Int, s1, s2, s3, s4, δ)
    δ2 = δ * δ
    δ3 = δ2 * δ
    return (s1 - n * δ, s2 - 2δ * s1 + n * δ2, s3 - 3δ * s2 + 3δ2 * s1 - n * δ3,
        s4 - 4δ * s3 + 6δ2 * s2 - 4δ3 * s1 + n * δ2 * δ2)
end

# The compensated form takes δ as a pair and carries every coefficient as a
# double-double, folding each product into the sums error-free, as
# `CoMoment`'s `recentre!` does.
# s + c·y, for c a pair and y an accumulator in s's storage.
@inline function addscaled(s::Compensated{A}, (h, l)::Tuple{A,A},
    y::Compensated{A}) where {A}
    p, e = twoprod(h, y.total)
    return compaddpair(s, p, e + (h * y.comp + l * y.total))
end
@inline addpair(s::Compensated, (h, l)) = compaddpair(s, h, l)
@inline ddneg((h, l)) = (-h, -l)
@inline function shiftpowers(n::Int, s1::Compensated{A}, s2::Compensated{A},
    s3::Compensated{A}, s4::Compensated{A}, δ::Tuple{A,A}) where {A}
    δ2 = ddmul(δ, δ)
    δ3 = ddmul(δ2, δ)
    δ4 = ddmul(δ2, δ2)
    r1 = addpair(s1, ddneg(ddscale(n, δ)))
    r2 = addpair(addscaled(s2, ddscale(-2, δ), s1), ddscale(n, δ2))
    r3 = addpair(addscaled(addscaled(s3, ddscale(-3, δ), s2), ddscale(3, δ2), s1),
        ddneg(ddscale(n, δ3)))
    r4 = addpair(
        addscaled(addscaled(addscaled(s4, ddscale(-4, δ), s3), ddscale(6, δ2), s2),
            ddscale(-4, δ3), s1),
        ddscale(n, δ4))
    return r1, r2, r3, r4
end

@inline function update!(st::CentralMomentState{N,C,T,M,S,W}, row) where {N,C,T,M,S,W}
    v = getproperty(row, C)
    if M && ismissing(v)
        st.missings += 1
        return nothing
    end
    x = convert(T, v)
    if !isfinite(x)
        st.nonfinite += 1
        return nothing
    end
    if st.n == 0
        st.k = x
    else
        st.age += 1
    end
    d = deviation(S, x, st.k)
    st.s1, st.s2, st.s3, st.s4 = foldpowers(st.s1, st.s2, st.s3, st.s4, d, 1)
    if W
        q = fourth(d)
        st.peak = ifelse(isfinite(q) & (q > st.peak), q, st.peak)
    end
    st.n += 1
    compensable(T) && 2 * st.age > st.n && recentre!(st, x)
    return nothing
end

@inline function downdate!(st::CentralMomentState{N,C,T,M,S,W},
    row) where {N,C,T,M,S,W}
    v = getproperty(row, C)
    if M && ismissing(v)
        st.missings -= 1
        return nothing
    end
    x = convert(T, v)
    if !isfinite(x)
        st.nonfinite -= 1
        return nothing
    end
    st.n -= 1
    if st.n == 0
        st.age = 0
        st.s1 = st.s2 = st.s3 = st.s4 = acczero(S)
        st.peak = zero(T)
    else
        d = deviation(S, x, st.k)
        st.s1, st.s2, st.s3, st.s4 = foldpowers(st.s1, st.s2, st.s3, st.s4, d, -1)
    end
    return nothing
end

# Read on every eviction, so the compensated pair is read as is. A Σd⁴ gone to
# zero under round-off asks too; a NaN one doesn't. The ratio is applied once
# to each side, since `REFOLDRATIO²` overflows Float16.
@inline function needsrefold(st::CentralMomentState{N,C,T,M,S,W}) where {N,C,T,M,S,W}
    W || return false
    r = convert(T, REFOLDRATIO)
    return st.peak / r > r * (st.s4.total + st.s4.comp)
end

@inline function recentre!(st::CentralMomentState{N,C,T,M,S}, k) where {N,C,T,M,S}
    st.s1, st.s2, st.s3, st.s4 =
        shiftpowers(st.n, st.s1, st.s2, st.s3, st.s4, deviation(S, k, st.k))
    st.k = k
    st.age = 0
    return nothing
end

# Reads every input before writing, since `dest` may alias either. b's sums
# are re-expressed at a's shift; an empty side contributes only its counts.
function combine!(dest::CentralMomentState{N,C,T,M,S,W},
    a::CentralMomentState{N,C,T,M,S,W},
    b::CentralMomentState{N,C,T,M,S,W}) where {N,C,T,M,S,W}
    nonfinite = a.nonfinite + b.nonfinite
    missings = a.missings + b.missings
    if b.n == 0
        n, age, k, s1, s2, s3, s4, peak = a.n, a.age, a.k, a.s1, a.s2, a.s3, a.s4, a.peak
    elseif a.n == 0
        n, age, k, s1, s2, s3, s4, peak = b.n, b.age, b.k, b.s1, b.s2, b.s3, b.s4, b.peak
    else
        b1, b2, b3, b4 = shiftpowers(b.n, b.s1, b.s2, b.s3, b.s4, deviation(S, a.k, b.k))
        s1, s2 = accmerge(a.s1, b1), accmerge(a.s2, b2)
        s3, s4 = accmerge(a.s3, b3), accmerge(a.s4, b4)
        n, age, k, peak = a.n + b.n, a.age + b.n, a.k, max(a.peak, b.peak)
    end
    dest.n, dest.age, dest.k = n, age, k
    dest.s1, dest.s2, dest.s3, dest.s4 = s1, s2, s3, s4
    dest.peak = peak
    dest.nonfinite, dest.missings = nonfinite, missings
    return nothing
end

# The centred sums from the shifted ones, with r = S1/n the mean's offset from
# the shift (so n·r = S1):
#   m2 = S2 - S1²/n
#   m3 = S3 - 3r·S2 + 2r²·S1
#   m4 = S4 - 4r·S3 + 6r²·S2 - 3r³·S1
# Each term sits at the scale of the distance D from the shift to the window,
# and the result at the spread s's, so a float evaluation would lose
# eps·(D/s)^k: up to 2e-12 of a four-row kurtosis whose shift was an older row.
# So compensated sums are read as double-doubles and the expansion is evaluated
# in double-double (`dd*` below), m2 by `CoMoment`'s own `comomentvalue`, so a
# flat window is exactly 0 wherever the variance's is. Plain storage evaluates
# it directly. m2 and m4 are clamped at 0.
@inline function centralvalue(n::Int, s1, s2, s3, s4)
    m2 = comomentvalue(n, s2, s1, s1)
    r = s1 / n
    m3 = s3 - r * (3 * s2 - 2 * r * s1)
    m4 = s4 - r * (4 * s3 - r * (6 * s2 - 3 * r * s1))
    return (max(m2, zero(m2)), m3, max(m4, zero(m4)))
end
@inline function centralvalue(n::Int, s1::Compensated{A}, s2::Compensated{A},
    s3::Compensated{A}, s4::Compensated{A}) where {A}
    compfinite(s1) && compfinite(s2) && compfinite(s3) && compfinite(s4) ||
        return centralvalue(n, compvalue(s1), compvalue(s2), compvalue(s3),
            compvalue(s4))
    m2 = comomentvalue(n, s2, s1, s1)
    a1, a2, a3, a4 = dd(s1), dd(s2), dd(s3), dd(s4)
    r = dddiv(a1, convert(A, n))
    # m3 = a3 - r·(3a2 - 2r·a1)
    m3 = ddsub(a3, ddmul(r, ddsub(ddscale(3, a2), ddscale(2, ddmul(r, a1)))))
    # m4 = a4 - r·(4a3 - r·(6a2 - 3r·a1))
    t = ddsub(ddscale(6, a2), ddscale(3, ddmul(r, a1)))
    t = ddsub(ddscale(4, a3), ddmul(r, t))
    m4 = ddsub(a4, ddmul(r, t))
    v3, v4 = m3[1] + m3[2], m4[1] + m4[2]
    return (max(m2, zero(m2)), v3, max(v4, zero(v4)))
end

# Minimal double-double arithmetic over pairs (hi, lo), for `deviation`,
# `shiftpowers` and `centralvalue`.
@inline dd(s::Compensated) = fasttwosum(s.total, s.comp)
@inline fasttwosum(a, b) = (s = a + b; (s, b - (s - a)))
@inline function twosum(a, b)
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)
end
@inline function ddadd((ah, al), (bh, bl))
    s, e = twosum(ah, bh)
    return fasttwosum(s, e + (al + bl))
end
@inline ddsub(a, (bh, bl)) = ddadd(a, (-bh, -bl))
@inline function ddmul((ah, al), (bh, bl))
    p, e = twoprod(ah, bh)
    return fasttwosum(p, e + (ah * bl + al * bh))
end
# k a small integer, so k·hi is formed exactly as a pair.
@inline function ddscale(k::Int, (h, l))
    kf = convert(typeof(h), k)
    p, e = twoprod(kf, h)
    return fasttwosum(p, e + kf * l)
end
@inline function dddiv((ah, al), d)
    q = ah / d
    p, e = twoprod(q, d)
    return fasttwosum(q, (((ah - p) - e) + al) / d)
end

# `missing` while a missing row is folded, else NaNs while a nonfinite one is.
# At most one row has no spread, exactly.
@inline function value(st::CentralMomentState{N,C,T,M}) where {N,C,T,M}
    V = comomenttype(T)
    R = maybemissing(NTuple{3,V}, M)
    M && st.missings > 0 && return NamedTuple{(N,),Tuple{R}}((missing,))
    if st.nonfinite > 0
        nan = convert(V, NaN)
        return NamedTuple{(N,),Tuple{R}}(((nan, nan, nan),))
    end
    st.n <= 1 && return NamedTuple{(N,),Tuple{R}}(((zero(V), zero(V), zero(V)),))
    m = centralvalue(st.n, st.s1, st.s2, st.s3, st.s4)
    return NamedTuple{(N,),Tuple{R}}((convert(NTuple{3,V}, m),))
end

function widenstate(st::CentralMomentState{N,C}, intypes::NamedTuple) where {N,C}
    w = centralfresh(CentralMoments{C}(), intypes[C])
    typeof(w) === typeof(st) && return st
    return centralwiden!(w, st)
end
function centralwiden!(w::CentralMomentState{N,C,T,M,S},
    st::CentralMomentState) where {N,C,T,M,S}
    w.n, w.age, w.k = st.n, st.age, convert(T, st.k)
    w.s1, w.s2 = widenacc(S, st.s1), widenacc(S, st.s2)
    w.s3, w.s4 = widenacc(S, st.s3), widenacc(S, st.s4)
    w.peak = convert(T, st.peak)
    w.nonfinite, w.missings = st.nonfinite, st.missings
    return w
end

withmissings!(w::CentralMomentState, n::Int) = (w.missings = n; w)

"""
    Skewness(column::ColumnSpec; corrected = true) -> Summarizer

The skewness of `column`, in `:{column}_skewness` (`missing` for no rows,
`NaN` if a value is NaN or ±Inf). With `corrected = true` it is the
sample-adjusted `G1 = g1·√(n(n-1))/(n-2)`, as pandas' rolling `skew` and
Excel's `SKEW` compute; with `false`, the moment estimator
`g1 = m₃/m₂^(3/2)` over population moments, as `StatsBase.skewness`. `NaN`
when every value is equal, and for fewer than three rows when corrected.
Integer input gives `Float64`. Like [`Variance`](@ref), the state folds
deviations from a shift near the data, so a large level doesn't cancel.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Keywords
- `corrected = true`: `G1` rather than `g1`. It is not part of the output
  name, so both forms of one column cannot be requested together.
"""
struct Skewness{C} <: GroupSummarizer
    corrected::Bool
end
Skewness(column::ColumnSpec; corrected::Bool = true) =
    withterms(Skewness{colname(column)}(corrected), column)

"""
    Kurtosis(column::ColumnSpec; corrected = true) -> Summarizer

The excess kurtosis of `column` (a normal distribution gives 0), in
`:{column}_kurtosis` (`missing` for no rows, `NaN` if a value is NaN or ±Inf).
With `corrected = true` it is the sample-adjusted
`G2 = (n-1)((n+1)·g2 + 6)/((n-2)(n-3))`, as pandas' rolling `kurt`, Excel's
`KURT` and TA-Lib's `KURTOSIS` compute; with `false`, the moment estimator
`g2 = m₄/m₂² - 3` over population moments, as `StatsBase.kurtosis`. `NaN`
when every value is equal, and for fewer than four rows when corrected.
Integer input gives `Float64`. Like [`Variance`](@ref), the state folds
deviations from a shift near the data, so a large level doesn't cancel.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Keywords
- `corrected = true`: `G2` rather than `g2`. It is not part of the output
  name, so both forms of one column cannot be requested together.
"""
struct Kurtosis{C} <: GroupSummarizer
    corrected::Bool
end
Kurtosis(column::ColumnSpec; corrected::Bool = true) =
    withterms(Kurtosis{colname(column)}(corrected), column)

# The shape statistic K (`:skewness` or `:kurtosis`) of the centred sums D,
# emitted as N; R is the corrected flag. Fieldless, like `CovarianceState`.
struct ShapeState{N,D,K,R} <: DerivedState end

for (S, K) in ((:Skewness, :skewness), (:Kurtosis, :kurtosis))
    @eval begin
        dependencies(::$S{C}) where {C} = (Count(), CentralMoments{C}())
        emptyvalue(::$S{C}) where {C} =
            NamedTuple{(Symbol(C, :_, $(QuoteNode(K))),)}((missing,))
        fresh(s::$S{C}, ::NamedTuple) where {C} =
            ShapeState{outname(s),centralname(C),$(QuoteNode(K)),s.corrected}()
    end
end

# The value type from the centred sums' declared field type: the triple's
# element type, `missing` kept.
shapetype(::Type{Missing}) = Missing
shapetype(::Type{F}) where {F} =
    maybemissing(fieldtype(nonmissingtype(F), 1), Missing <: F)

# The statistic from the centred sums over n rows. No spread (m2 not above 0,
# which a NaN isn't either) gives NaN, as do too few rows for the correction.
@inline shapeof(::Missing, ::Int, ::Val, ::Val) = missing
@inline function shapeof((m2, m3, m4)::NTuple{3,V}, n::Int, ::Val{:skewness},
    ::Val{R}) where {V,R}
    (m2 > 0 && (!R || n > 2)) || return convert(V, NaN)
    g1 = sqrt(convert(V, n)) * (m3 / m2) / sqrt(m2)
    R || return g1
    return g1 * sqrt(convert(V, n) * (n - 1)) / (n - 2)
end
@inline function shapeof((m2, m3, m4)::NTuple{3,V}, n::Int, ::Val{:kurtosis},
    ::Val{R}) where {V,R}
    (m2 > 0 && (!R || n > 3)) || return convert(V, NaN)
    g2 = n * (m4 / m2) / m2 - 3
    R || return g2
    return (n - 1) * ((n + 1) * g2 + 6) / ((n - 2) * (n - 3))
end

@inline function value(::ShapeState{N,D,K,R}, vals::NamedTuple) where {N,D,K,R}
    V = shapetype(fieldtype(typeof(vals), D))
    return NamedTuple{(N,),Tuple{V}}((shapeof(vals[D], vals.count, Val(K), Val(R)),))
end
