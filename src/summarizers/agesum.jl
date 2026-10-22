"""
    AgeWeightedSum(column::ColumnSpec) -> Summarizer

The sum of `column` weighted by each row's age, `Σₖ k·yₖ` where `k` counts the
rows folded after row `k` (`0` for the newest), in `:{column}_ageweightedsum`
(`0` for no rows). With [`Sum`](@ref) and [`Count`](@ref) it gives the linearly
weighted moving average `(n·Σy − Σk·y) / (n(n+1)/2)`, whose newest row weighs
`n`, and a least-squares fit on the row number.

The element type follows [`Sum`](@ref), and floats use compensated summation
with NaN and ±Inf counted separately, so a rolling window recovers once they
leave. The newest row weighs `0`, so a NaN or ±Inf there contributes nothing
until a later row arrives; a `missing` anywhere gives `missing`.

# Arguments
- `column`: the column to weight, or a row term `name => f` (see [`ColumnSpec`](@ref)).
  Age is counted in rows, in stream order, so
  rows tied in time have different ages.

```jldoctest
df = DataFrame(time = 1:4, y = [1.0, 2.0, 3.0, 4.0])
p = readtable(df) |> addsummarycolumns(AgeWeightedSum(:y))
DataFrame(load(Context(0, 10), p))

# output

4×3 DataFrame
 Row │ time   y        y_ageweightedsum
     │ Int64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1      1.0               0.0
   2 │     2      2.0               1.0
   3 │     3      3.0               4.0
   4 │     4      4.0              10.0
```
"""
struct AgeWeightedSum{C} <: GroupSummarizer end
AgeWeightedSum(column::ColumnSpec) = withterms(AgeWeightedSum{colname(column)}(), column)

# The state folds the row count n, S1 = Σy and S2 = Σk·y. A new row ages every
# folded row by one, adding S1 to S2, then joins S1 at weight 0. Removing the
# oldest row (the only one `downdate!` is handed) takes back its weight n - 1.
# Combining ages a's rows by b's count. n counts every row, missing and
# nonfinite included, since age is counted in rows. A `missing` input is
# counted, not folded, under the flag M, as in `AccumState`.
mutable struct AgeSumState{N,C,A,M} <: SummarizerState
    n::Int
    s1::A
    s2::A
    missings::Int
end

# The float form keeps S1 as a `Compensated`, whose counters classify each NaN
# and ±Inf input. S2 keeps only the finite pair (zero counters) and takes its
# nonfinite classification from S1's, minus the newest row's, whose weight is 0
# (`newest`: 0 finite or missing, 1 NaN, 2 +Inf, 3 -Inf).
#
# As in the sum family (issue #97), a compensated S1 loses what lies far below
# a dominant row's round-off, and S2, which adds S1 on every row, folds the
# loss in again row after row, so it stays after the row has gone. So the
# windowed state (`freshwindowed`, flag W) keeps the largest finite |y|
# folded (`peak`) and a plain live Σ|y| (`mag`), and asks to be refolded once
# peak exceeds REFOLDRATIO times mag. The ordinary state never downdates, so
# it tracks nothing (both stay 0).
mutable struct CompensatedAgeSumState{N,C,A<:AbstractFloat,M,W} <: SummarizerState
    n::Int
    s1::Compensated{A}
    s2::Compensated{A}
    peak::A
    mag::A
    newest::Int8
    missings::Int
end

agesumname(C::Symbol) = Symbol(C, :_ageweightedsum)

emptyvalue(::AgeWeightedSum{C}) where {C} = NamedTuple{(agesumname(C),)}((0,))
fresh(::AgeWeightedSum{C}, intypes::NamedTuple) where {C} =
    agesumfresh(Val(agesumname(C)), Val(C), sumtype(intypes[C]))
freshwindowed(::AgeWeightedSum{C}, intypes::NamedTuple) where {C} =
    agesumfresh(Val(agesumname(C)), Val(C), sumtype(intypes[C]), Val(true))

# The state for accumulator type A, chosen as `accumfresh` chooses: the fold is
# at the non-missing type, compensated for a fixed-precision float (tracking
# for a refold when windowed), and an all-Missing column gets a `MissingState`.
function agesumfresh(::Val{N}, ::Val{C}, ::Type{A},
    ::Val{W} = Val(false)) where {N,C,A,W}
    A === Missing &&
        return MissingState{N}(AgeWeightedSum{C}())
    M = Missing <: A
    An = nonmissingtype(A)
    compensable(An) && return CompensatedAgeSumState{N,C,An,M,W}()
    return AgeSumState{N,C,An,M}()
end

AgeSumState{N,C,A,M}() where {N,C,A,M} =
    AgeSumState{N,C,A,M}(0, convert(A, 0), convert(A, 0), 0)
CompensatedAgeSumState{N,C,A,M,W}() where {N,C,A,M,W} =
    CompensatedAgeSumState{N,C,A,M,W}(0, compzero(A), compzero(A), zero(A), zero(A),
        Int8(0), 0)
fresh(::S) where {S<:Union{AgeSumState,CompensatedAgeSumState}} = S()
@inline function fresh!(st::AgeSumState{N,C,A}) where {N,C,A}
    st.n = 0
    st.s1 = convert(A, 0)
    st.s2 = convert(A, 0)
    st.missings = 0
    return st
end
@inline function update!(st::AgeSumState{N,C,A}, row) where {N,C,A}
    v = getproperty(row, C)
    st.s2 += st.s1
    st.n += 1
    ismissing(v) ? (st.missings += 1) : (st.s1 += convert(A, v))
    return nothing
end
@inline function downdate!(st::AgeSumState{N,C,A}, row) where {N,C,A}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings -= 1
    else
        x = convert(A, v)
        st.s2 -= convert(A, st.n - 1) * x
        st.s1 -= x
    end
    st.n -= 1
    return nothing
end
function combine!(dest::AgeSumState{N,C,A,M}, a::AgeSumState{N,C,A,M},
    b::AgeSumState{N,C,A,M}) where {N,C,A,M}
    n = a.n + b.n
    s1 = a.s1 + b.s1
    s2 = a.s2 + a.s1 * convert(A, b.n) + b.s2
    missings = a.missings + b.missings
    dest.n, dest.s1, dest.s2, dest.missings = n, s1, s2, missings
    return nothing
end
value(st::AgeSumState{N,C,A,M}) where {N,C,A,M} =
    NamedTuple{(N,),Tuple{maybemissing(A, M)}}((
        M && st.missings > 0 ? missing : st.s2,))
# Carry a plain state's numbers into a wider fresh one. A plain state tracks no
# nonfinite counts, so its totals enter a compensated one as finite terms.
function agesumfrom(w::AgeSumState{N,C,A,M}, n, s1, s2, missings) where {N,C,A,M}
    return AgeSumState{N,C,A,M}(n, convert(A, s1), convert(A, s2), missings)
end
function agesumfrom(w::CompensatedAgeSumState{N,C,A,M,W}, n, s1, s2,
    missings) where {N,C,A,M,W}
    return CompensatedAgeSumState{N,C,A,M,W}(n, widenacc(Compensated{A}, s1),
        widenacc(Compensated{A}, s2), zero(A), zero(A), Int8(0), missings)
end

@inline agesumclass(x::AbstractFloat) =
    isfinite(x) ? Int8(0) : isnan(x) ? Int8(1) : x > 0 ? Int8(2) : Int8(3)

@inline function fresh!(st::CompensatedAgeSumState{N,C,A}) where {N,C,A}
    st.n = 0
    st.s1 = compzero(A)
    st.s2 = compzero(A)
    st.peak = st.mag = zero(A)
    st.newest = Int8(0)
    st.missings = 0
    return st
end
@inline function update!(st::CompensatedAgeSumState{N,C,A,M,W},
    row) where {N,C,A,M,W}
    v = getproperty(row, C)
    st.s2 = pairadd(st.s2, st.s1.total, st.s1.comp)
    st.n += 1
    if ismissing(v)
        st.missings += 1
        st.newest = Int8(0)
    else
        x = convert(A, v)
        st.s1 = compadd(st.s1, x)
        st.newest = agesumclass(x)
        if W && isfinite(x)
            a = abs(x)
            st.peak = ifelse(a > st.peak, a, st.peak)  # finite: no NaN to order
            st.mag += a
        end
    end
    return nothing
end
@inline function downdate!(st::CompensatedAgeSumState{N,C,A,M,W},
    row) where {N,C,A,M,W}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings -= 1
    else
        x = convert(A, v)
        isfinite(x) && (st.s2 = pairsubprod(st.s2, convert(A, st.n - 1), x))
        st.s1 = compsub(st.s1, x)
        W && isfinite(x) && (st.mag -= abs(x))
    end
    st.n -= 1
    st.n == 0 && (st.newest = Int8(0))
    return nothing
end
# A mag gone negative or to zero under round-off asks too.
@inline needsrefold(st::CompensatedAgeSumState{N,C,A,M,W}) where {N,C,A,M,W} =
    W && st.peak > convert(A, REFOLDRATIO) * st.mag
function combine!(dest::CompensatedAgeSumState{N,C,A,M,W},
    a::CompensatedAgeSumState{N,C,A,M,W},
    b::CompensatedAgeSumState{N,C,A,M,W}) where {N,C,A,M,W}
    k = convert(A, b.n)
    p, e = twoprod(a.s1.total, k)
    s2 = pairadd(pairadd(a.s2, p, e + a.s1.comp * k), b.s2.total, b.s2.comp)
    s1 = compmerge(a.s1, b.s1)
    newest = b.n > 0 ? b.newest : a.newest
    n = a.n + b.n
    missings = a.missings + b.missings
    dest.n, dest.s1, dest.s2, dest.newest, dest.missings = n, s1, s2, newest, missings
    dest.peak, dest.mag = max(a.peak, b.peak), a.mag + b.mag
    return nothing
end
# S2's nonfinite terms are S1's less the newest row's, whose weight is 0: a NaN
# or ±Inf joins S2 only once a later row has aged it.
@inline function agesumvalue(st::CompensatedAgeSumState{N,C,A}) where {N,C,A}
    s1 = st.s1
    return compvalue(
        Compensated{A}(st.s2.total, st.s2.comp,
            s1.nans - (st.newest == 1), s1.posinf - (st.newest == 2),
            s1.neginf - (st.newest == 3)),
    )
end
value(st::CompensatedAgeSumState{N,C,A,M}) where {N,C,A,M} =
    NamedTuple{(N,),Tuple{maybemissing(A, M)}}((
        M && st.missings > 0 ? missing : agesumvalue(st),))
function widenstate(st::Union{AgeSumState{N,C},CompensatedAgeSumState{N,C}},
    intypes::NamedTuple) where {N,C}
    w = agesumfresh(Val(N), Val(C), sumtype(intypes[C]))
    typeof(w) === typeof(st) && return st
    return agesumwiden(w, st)
end
agesumwiden(w, st::AgeSumState) = agesumfrom(w, st.n, st.s1, st.s2, st.missings)
function agesumwiden(w::CompensatedAgeSumState{N,C,A2,M2,W},
    st::CompensatedAgeSumState) where {N,C,A2,M2,W}
    return CompensatedAgeSumState{N,C,A2,M2,W}(st.n, widencomp(A2, st.s1),
        widencomp(A2, st.s2), convert(A2, st.peak), convert(A2, st.mag),
        st.newest, st.missings)
end
# To an arbitrary-precision float: the plain state, from the IEEE values.
agesumwiden(w::AgeSumState, st::CompensatedAgeSumState) =
    agesumfrom(w, st.n, compvalue(st.s1), agesumvalue(st), st.missings)

# n missing rows age the (zero) sums and leave `newest` at 0.
withmissings!(w::Union{AgeSumState,CompensatedAgeSumState}, n::Int) =
    (w.n = n; w.missings = n; w)
