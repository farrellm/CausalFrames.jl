# The dependent moment summarizers: Moment, Mean, Variance, Std, Covariance and
# Correlation, read back from the sum family and the co-moments.

"""
    Moment(column::ColumnSpec, n::Integer) -> Summarizer

The `n`-th raw moment of `column`, the mean of `column ^ n`, in
`:{column}_moment_{n}` (`missing` for no rows). Computed from
[`SumPower`](@ref)`(column, n)` and [`Count`](@ref); integer input gives
`Float64`.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
- `n`: the order, part of the output name.
"""
struct Moment{C} <: GroupSummarizer
    order::Int
end
Moment(column::ColumnSpec, order::Integer) =
    withterms(Moment{colname(column)}(Int(order)), column)

# A sum D over the row count, emitted as N: Moment's power sum or Mean's plain
# sum. Fieldless; the names are type parameters so the two-argument `value`
# infers.
struct CountRatioState{N,D} <: DerivedState end

# The value type comes from the dependencies' declared field types, not
# `typeof` of the quotient, so a missing sum keeps the Union{Missing,...} eltype.
# `promote_op` is behind a function of the types: written inline here, Julia
# 1.10 cannot fold it to a constant for a Missing-admitting sum, and the whole
# summary row's type widens.
ratiotype(::Type{A}, ::Type{B}) where {A,B} = Base.promote_op(/, A, B)
@inline function value(::CountRatioState{N,D}, vals::NamedTuple) where {N,D}
    V = ratiotype(fieldtype(typeof(vals), D), fieldtype(typeof(vals), :count))
    return NamedTuple{(N,),Tuple{V}}((vals[D] / vals.count,))
end

dependencies(m::Moment{C}) where {C} = (Count(), SumPower(C, m.order))
emptyvalue(m::Moment{C}) where {C} =
    NamedTuple{(Symbol(C, :_moment_, m.order),)}((missing,))
fresh(m::Moment{C}, ::NamedTuple) where {C} =
    CountRatioState{outname(m),outname(SumPower(C, m.order))}()

"""
    Mean(column::ColumnSpec) -> Summarizer

The mean of `column`, in `:{column}_mean` (`missing` for no rows). Computed from
[`Sum`](@ref)`(column)` and [`Count`](@ref); integer input gives `Float64`,
`Float32` stays `Float32`.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Mean{C} <: GroupSummarizer end
Mean(column::ColumnSpec) = withterms(Mean{colname(column)}(), column)

dependencies(::Mean{C}) where {C} = (Count(), Sum(C))
emptyvalue(::Mean{C}) where {C} = NamedTuple{(Symbol(C, :_mean),)}((missing,))
fresh(m::Mean{C}, ::NamedTuple) where {C} =
    CountRatioState{outname(m),outname(Sum(C))}()

"""
    Variance(column::ColumnSpec; corrected = true) -> Summarizer

The variance of `column`, as `Statistics.var`, in `:{column}_variance`
(`missing` for no rows, `NaN` if a value is NaN or ±Inf). Integer input gives
`Float64`. The state folds deviations from a shift near the data, so a large
level beside a small spread doesn't cancel, and round-off never makes the
variance negative.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Keywords
- `corrected = true`: divide by `n - 1` (a single row gives `NaN`); `false`
  divides by `n`. It is not part of the output name, so both forms of one column
  cannot be requested together.
"""
struct Variance{C} <: GroupSummarizer
    corrected::Bool
end
Variance(column::ColumnSpec; corrected::Bool = true) =
    withterms(Variance{colname(column)}(corrected), column)

dependencies(::Variance{C}) where {C} = (Count(), CoMoment{C,C}())
emptyvalue(::Variance{C}) where {C} =
    NamedTuple{(Symbol(C, :_variance),)}((missing,))
# A variance is the covariance of a column with itself, so it uses
# CovarianceState (below) over the co-moment of the column with itself.
fresh(v::Variance{C}, ::NamedTuple) where {C} =
    CovarianceState{outname(v),outname(CoMoment{C,C}()),v.corrected}()

"""
    Std(column::ColumnSpec; corrected = true) -> Summarizer

The standard deviation of `column`, as `Statistics.std`, in `:{column}_std`
(`missing` for no rows): the square root of [`Variance`](@ref), with round-off
negatives clamped to zero.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Keywords
- `corrected = true`: as for [`Variance`](@ref).
"""
struct Std{C} <: GroupSummarizer
    corrected::Bool
end
Std(column::ColumnSpec; corrected::Bool = true) =
    withterms(Std{colname(column)}(corrected), column)

struct StdState{N,V} <: DerivedState end

stdsqrt(::Missing) = missing
stdsqrt(v) = sqrt(max(v, zero(v)))

dependencies(s::Std{C}) where {C} = (Variance(C; corrected = s.corrected),)
emptyvalue(::Std{C}) where {C} = NamedTuple{(Symbol(C, :_std),)}((missing,))
fresh(s::Std{C}, ::NamedTuple) where {C} =
    StdState{outname(s),outname(Variance(C))}()
@inline function value(::StdState{N,V}, vals::NamedTuple) where {N,V}
    T = Base.promote_op(sqrt, fieldtype(typeof(vals), V))
    return NamedTuple{(N,),Tuple{T}}((stdsqrt(vals[V]),))
end

"""
    Covariance(a::ColumnSpec, b::ColumnSpec; corrected = true) -> Summarizer

The covariance of `a` and `b`, as `Statistics.cov`, in `:{a}_{b}_covariance`
(`missing` for no rows, `NaN` if a value is NaN or ±Inf). Like
[`Variance`](@ref), it folds deviations from a shift near the data, so large
levels don't cancel.

# Arguments
- `a`, `b`: column names or row terms `name => f` (see [`ColumnSpec`](@ref)).

# Keywords
- `corrected = true`: as for [`Variance`](@ref), including that it is not part
  of the output name.
"""
struct Covariance{A,B} <: GroupSummarizer
    corrected::Bool
end
Covariance(a::ColumnSpec, b::ColumnSpec; corrected::Bool = true) =
    withterms(Covariance{colname(a),colname(b)}(corrected), a, b)

# The co-moment D over the divisor `n - Int(R)`, emitted as N. R (the
# corrected flag) is a type parameter so the state stays fieldless. Variance
# uses it too, over the co-moment of a column with itself.
struct CovarianceState{N,D,R} <: DerivedState end

covname(a, b) = Symbol(a, :_, b, :_covariance)
# Reads the canonically ordered co-moment: Covariance(:y, :x) produces
# :y_x_covariance from the shared :x_y_comoment accumulator.
dependencies(::Covariance{A,B}) where {A,B} =
    (Count(), canonicalcomoment(A, B))
emptyvalue(::Covariance{A,B}) where {A,B} =
    NamedTuple{(covname(A, B),)}((missing,))
fresh(c::Covariance{A,B}, ::NamedTuple) where {A,B} =
    CovarianceState{outname(c),outname(canonicalcomoment(A, B)),c.corrected}()
@inline function value(::CovarianceState{N,D,R}, vals::NamedTuple) where {N,D,R}
    V = ratiotype(fieldtype(typeof(vals), D), Int)
    return NamedTuple{(N,),Tuple{V}}((vals[D] / (vals.count - Int(R)),))
end

"""
    Correlation(a::ColumnSpec, b::ColumnSpec) -> Summarizer

The Pearson correlation of `a` and `b`, as `Statistics.cor`, clamped to
`[-1, 1]`, in `:{a}_{b}_correlation` (`missing` for no rows, `NaN` for one).
Computed from [`Covariance`](@ref) and the two columns' [`Std`](@ref)s; there is
no `corrected` keyword, since the correction cancels.

# Arguments
- `a`, `b`: column names or row terms `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Correlation{A,B} <: GroupSummarizer end
Correlation(a::ColumnSpec, b::ColumnSpec) =
    withterms(Correlation{colname(a),colname(b)}(), a, b)

struct CorrelationState{N,CV,SA,SB} <: DerivedState end

clampcor(::Missing) = missing
clampcor(x) = clamp(x, -one(x), one(x))

corname(a, b) = Symbol(a, :_, b, :_correlation)
dependencies(::Correlation{A,B}) where {A,B} =
    (Covariance(A, B), Std(A), Std(B))
emptyvalue(::Correlation{A,B}) where {A,B} =
    NamedTuple{(corname(A, B),)}((missing,))
fresh(c::Correlation{A,B}, ::NamedTuple) where {A,B} =
    CorrelationState{outname(c),outname(Covariance(A, B)),outname(Std(A)),
        outname(Std(B))}()
@inline function value(::CorrelationState{N,CV,SA,SB},
    vals::NamedTuple) where {N,CV,SA,SB}
    Cvf = fieldtype(typeof(vals), CV)
    Saf = fieldtype(typeof(vals), SA)
    Sbf = fieldtype(typeof(vals), SB)
    V = Base.promote_op(/, Cvf, Base.promote_op(*, Saf, Sbf))
    return NamedTuple{(N,),Tuple{V}}((clampcor(vals[CV] / (vals[SA] * vals[SB])),))
end
