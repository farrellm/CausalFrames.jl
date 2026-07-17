# The summarizer interface and the concrete summarizers. A summarization is
# split in two: an immutable Summarizer holding only the configuration, and a
# SummarizerState holding the running state. The state is built from the input
# columns' element types, so its value fields — and hence the element types of
# the columns it produces — are concrete. That split is what makes an output
# column's type a consequence of the input schema rather than an accident of
# the values, and it is what lets the folding loops in summarize.jl run
# type-stable behind a function barrier.

"""
    Summarizer

Abstract supertype for summarization *configurations*. A concrete summarizer
is immutable and holds only configuration — typically the column to summarize,
carried as a *type parameter* so that the output column names it implies are
known to the compiler. It implements:

- [`emptyvalue`](@ref)`(s)` — the summary of no rows, as a `NamedTuple` whose
  keys are the output column names;
- [`fresh`](@ref)`(s, intypes)` — a zero [`SummarizerState`](@ref), typed for
  input columns whose element types are given by `intypes`.

Output column names are deterministic, formed by suffixing the column name
(e.g. `Sum(:x)` produces `:x_sum`); summarizers with identical output names
are treated as identical and share state.

A summarizer may depend on the values of other summarizers by implementing
[`dependencies`](@ref)`(s)` and the two-argument form of [`value`](@ref);
dependencies are resolved through the same name-keyed deduplication, so
shared work is computed once, and appear in the output only when requested
by the user themselves.

Planned refinements will add structured subtypes: a monoid subtype (mergeable
state, enabling map-reduce) and a group subtype (invertible updates, enabling
efficient rolling windows).
"""
abstract type Summarizer end

"""
    SummarizerState

Abstract supertype for the running state of one summarization, built by
[`fresh`](@ref)`(s, intypes)` from a [`Summarizer`](@ref) and the input
columns' element types. Because the state's value fields are concrete, so are
the columns it produces. It implements:

- [`fresh`](@ref)`(st)` — a new state of the *same type* with zero state;
- [`update!`](@ref)`(st, row)` — fold one row into the state;
- [`value`](@ref)`(st)` — the current summary as a `NamedTuple` whose keys are
  the output column names;
- [`widenstate`](@ref)`(st, intypes)` — optionally, a state rebuilt for
  widened input columns.
"""
abstract type SummarizerState end

"""
    emptyvalue(s::Summarizer) -> NamedTuple

The summary of no rows, keyed by output column name. This is the only summary
available when the input has no rows at all: the chunk protocol never yields
an empty chunk, so an empty input carries no schema and no state can be built
for it. It is also where the summarization transforms read a summarizer's
output column names, before any data has been seen.
"""
function emptyvalue end

"""
    fresh(s::Summarizer, intypes::NamedTuple) -> SummarizerState
    fresh(st::SummarizerState) -> SummarizerState

A zero state. The first form builds one from a summarizer and the input
columns' element types, mirroring the row access in [`update!`](@ref):
`update!` reads `row[column]` where `fresh` reads `intypes[column]`. The
second form produces a new state of the same concrete type as `st`, which is
how the transforms obtain per-key-group and per-cycle states without
re-consulting the schema.

The summarization transforms treat the summarizers they are given as
prototypes, so one prototype serves many key groups and the caller's instance
is never mutated.
"""
function fresh end

"""
    update!(st::SummarizerState, row)

Fold one row into the state. `row` is a map-like row object supporting
`row.name` and `row[:name]` access, including `row.time`, so a summarizer may
read whichever columns it needs.
"""
function update! end

"""
    value(st::SummarizerState) -> NamedTuple
    value(st::SummarizerState, vals::NamedTuple) -> NamedTuple

The current summary. A summarizer may produce several values; the keys of the
returned `NamedTuple` are the output column names, and its value types are the
element types of the columns produced.

The two-argument form receives in `vals` the already-computed values of every
summarizer earlier in topological order — in particular of everything named by
[`dependencies`](@ref). It defaults to calling the one-argument form; a
dependent summarizer implements the two-argument form instead and may omit
the one-argument form entirely.

Only ever called on a state that has folded at least one row — the summary of
no rows is [`emptyvalue`](@ref).
"""
function value end

value(st::SummarizerState, ::NamedTuple) = value(st)

"""
    widenstate(st::SummarizerState, intypes::NamedTuple) -> SummarizerState

A state equivalent to `st` but typed for input columns of the element types in
`intypes`, carrying the accumulated value over. A source may infer a column's
element type per chunk (`readcsv` does), so a column can be `Int` in one chunk
and `Float64` in the next; the transforms promote the types they have seen and
call `widenstate` when that promotion changes something.

Defaults to returning `st` unchanged, which is correct for any state whose
type does not depend on the input, and which lets a summarizer opt out.
"""
widenstate(st::SummarizerState, ::NamedTuple) = st

"""
    dependencies(s::Summarizer) -> Tuple

The summarizers whose values `s` reads in the two-argument form of
[`value`](@ref). The summarization transforms expand dependencies —
recursively, in topological order — into the set of summarizers they fold,
deduplicated by output name, so a dependency equal to a user-requested
summarizer is computed once. Dependencies appear in the output only when the
user requested them themselves.

Defaults to `()`, which is correct for any self-contained summarizer.
"""
dependencies(::Summarizer) = ()

# The element type Base.sum produces over a column of eltype T: small signed
# and unsigned integers widen to Int/UInt, everything else keeps its type. The
# accumulator is built at this width up front, so the folding loop is a plain
# `+` that is both type-stable and immune to the overflow that accumulating in
# the input's own type would risk.
sumtype(::Type{T}) where {T} = Base.promote_op(Base.add_sum, T, T)

# The analogous widths for a product and for a dot product. `prodtype` is the
# element type `Base.prod` produces (small ints widen through `mul_prod` just
# as they do through `add_sum`); `dottype` is `sumtype` applied to the type of
# one `a * b` term, since a dot product is a sum of products.
prodtype(::Type{T}) where {T} = Base.promote_op(Base.mul_prod, T, T)
dottype(::Type{Ta}, ::Type{Tb}) where {Ta,Tb} =
    sumtype(Base.promote_op(*, Ta, Tb))

"""
    Count() -> Summarizer

Counts rows. Produces the output column `:count`, of type `Int`.
"""
struct Count <: Summarizer end

mutable struct CountState <: SummarizerState
    n::Int
end

emptyvalue(::Count) = (; count = 0)
fresh(::Count, ::NamedTuple) = CountState(0)
fresh(::CountState) = CountState(0)
@inline update!(st::CountState, row) = (st.n += 1; nothing)
value(st::CountState) = (; count = st.n)

"""
    Sum(column) -> Summarizer

Sums `column`. Produces the output column `Symbol(column, :_sum)`, e.g.
`Sum(:x)` produces `:x_sum`. The sum of no rows is `0`.

The output column's element type is the one `Base.sum` would produce: small
signed and unsigned integers widen (`Int32` sums to `Int64`), everything else
keeps its type (`Float32` sums to `Float32`).
"""
struct Sum{C} <: Summarizer end
Sum(column::Symbol) = Sum{column}()

mutable struct SumState{C,N,A} <: SummarizerState
    total::A
end

emptyvalue(::Sum{C}) where {C} = NamedTuple{(Symbol(C, :_sum),)}((0,))
function fresh(::Sum{C}, intypes::NamedTuple) where {C}
    A = sumtype(intypes[C])
    return SumState{C,Symbol(C, :_sum),A}(convert(A, 0))
end
fresh(::SumState{C,N,A}) where {C,N,A} = SumState{C,N,A}(convert(A, 0))
@inline update!(st::SumState{C}, row) where {C} =
    (st.total += getproperty(row, C); nothing)
value(st::SumState{C,N,A}) where {C,N,A} = NamedTuple{(N,),Tuple{A}}((st.total,))
function widenstate(st::SumState{C,N,A}, intypes::NamedTuple) where {C,N,A}
    A2 = sumtype(intypes[C])
    A2 === A && return st
    return SumState{C,N,A2}(convert(A2, st.total))
end

"""
    SumPower(column, n) -> Summarizer

Sums `column` raised to the power `n`. Produces the output column
`Symbol(column, :_sumpower_, n)`, e.g. `SumPower(:x, 2)` produces
`:x_sumpower_2`. The sum of no rows is `0`. The output column's element type
follows the same rule as [`Sum`](@ref), applied to the type of `column ^ n`.

`SumPower(column, 1)` produces `:x_sumpower_1`, a distinct column from
`Sum(:x)`'s `:x_sum`.
"""
struct SumPower{C} <: Summarizer
    power::Int
end
SumPower(column::Symbol, power::Integer) = SumPower{column}(Int(power))

mutable struct SumPowerState{C,N,A} <: SummarizerState
    power::Int
    total::A
end

powertype(::Type{T}, ::Int) where {T} = sumtype(Base.promote_op(^, T, Int))

emptyvalue(s::SumPower{C}) where {C} =
    NamedTuple{(Symbol(C, :_sumpower_, s.power),)}((0,))
function fresh(s::SumPower{C}, intypes::NamedTuple) where {C}
    A = powertype(intypes[C], s.power)
    return SumPowerState{C,Symbol(C, :_sumpower_, s.power),A}(s.power,
                                                              convert(A, 0))
end
fresh(st::SumPowerState{C,N,A}) where {C,N,A} =
    SumPowerState{C,N,A}(st.power, convert(A, 0))
@inline update!(st::SumPowerState{C}, row) where {C} =
    (st.total += getproperty(row, C)^st.power; nothing)
value(st::SumPowerState{C,N,A}) where {C,N,A} =
    NamedTuple{(N,),Tuple{A}}((st.total,))
function widenstate(st::SumPowerState{C,N,A}, intypes::NamedTuple) where {C,N,A}
    A2 = powertype(intypes[C], st.power)
    A2 === A && return st
    return SumPowerState{C,N,A2}(st.power, convert(A2, st.total))
end

"""
    Product(column) -> Summarizer

Multiplies `column`. Produces the output column `Symbol(column, :_product)`,
e.g. `Product(:x)` produces `:x_product`. The product of no rows is `1`.

The output column's element type is the one `Base.prod` would produce: small
signed and unsigned integers widen (`Int32` multiplies to `Int64`), everything
else keeps its type (`Float32` stays `Float32`). Like [`Sum`](@ref), the
accumulator is built at that width up front.
"""
struct Product{C} <: Summarizer end
Product(column::Symbol) = Product{column}()

mutable struct ProductState{C,N,A} <: SummarizerState
    total::A
end

emptyvalue(::Product{C}) where {C} = NamedTuple{(Symbol(C, :_product),)}((1,))
function fresh(::Product{C}, intypes::NamedTuple) where {C}
    A = prodtype(intypes[C])
    return ProductState{C,Symbol(C, :_product),A}(convert(A, 1))
end
fresh(::ProductState{C,N,A}) where {C,N,A} = ProductState{C,N,A}(convert(A, 1))
@inline update!(st::ProductState{C}, row) where {C} =
    (st.total *= getproperty(row, C); nothing)
value(st::ProductState{C,N,A}) where {C,N,A} = NamedTuple{(N,),Tuple{A}}((st.total,))
function widenstate(st::ProductState{C,N,A}, intypes::NamedTuple) where {C,N,A}
    A2 = prodtype(intypes[C])
    A2 === A && return st
    return ProductState{C,N,A2}(convert(A2, st.total))
end

"""
    DotProduct(a, b) -> Summarizer

Sums the elementwise product of columns `a` and `b`. Produces the output
column `Symbol(a, :_, b, :_dotproduct)`, e.g. `DotProduct(:x, :y)` produces
`:x_y_dotproduct`. The dot product of no rows is `0`.

The output column's element type is `Base.sum` applied to the type of `a * b`,
so it widens the same way [`Sum`](@ref) does. Each term is formed in the
accumulator's (widened) type, so a per-row product cannot overflow the way
multiplying in the input columns' own types would.
"""
struct DotProduct{A,B} <: Summarizer end
DotProduct(a::Symbol, b::Symbol) = DotProduct{a,b}()

mutable struct DotProductState{A,B,N,Acc} <: SummarizerState
    total::Acc
end

dotname(a, b) = Symbol(a, :_, b, :_dotproduct)
emptyvalue(::DotProduct{A,B}) where {A,B} = NamedTuple{(dotname(A, B),)}((0,))
function fresh(::DotProduct{A,B}, intypes::NamedTuple) where {A,B}
    Acc = dottype(intypes[A], intypes[B])
    return DotProductState{A,B,dotname(A, B),Acc}(convert(Acc, 0))
end
fresh(::DotProductState{A,B,N,Acc}) where {A,B,N,Acc} =
    DotProductState{A,B,N,Acc}(convert(Acc, 0))
@inline update!(st::DotProductState{A,B,N,Acc}, row) where {A,B,N,Acc} =
    (st.total += convert(Acc, getproperty(row, A)) * convert(Acc, getproperty(row, B));
     nothing)
value(st::DotProductState{A,B,N,Acc}) where {A,B,N,Acc} =
    NamedTuple{(N,),Tuple{Acc}}((st.total,))
function widenstate(st::DotProductState{A,B,N,Acc}, intypes::NamedTuple) where {A,B,N,Acc}
    Acc2 = dottype(intypes[A], intypes[B])
    Acc2 === Acc && return st
    return DotProductState{A,B,N,Acc2}(convert(Acc2, st.total))
end

"""
    Moment(column, n) -> Summarizer

The `n`-th raw moment of `column`: the mean of `column ^ n`. Produces the
output column `Symbol(column, :_moment_, n)`, e.g. `Moment(:x, 2)` produces
`:x_moment_2`; `Moment(:x, 1)` is the mean. The moment of no rows is
`missing`.

A dependent summarizer, computed as `SumPower(column, n)` divided by
[`Count`](@ref) — those are folded alongside it but appear in the output only
if requested themselves. The output column's element type is the division's
result (`Int` input divides to `Float64`, `Float32` stays `Float32`).
"""
struct Moment{C} <: Summarizer
    order::Int
end
Moment(column::Symbol, order::Integer) = Moment{column}(Int(order))

# The state is fieldless: the moment is derived entirely from its
# dependencies' values at emission time, so its own name N and the power
# sum's name D are all it needs, baked as type parameters so the two-argument
# `value` infers.
struct MomentState{C,N,D} <: SummarizerState end

dependencies(m::Moment{C}) where {C} = (Count(), SumPower(C, m.order))
emptyvalue(m::Moment{C}) where {C} =
    NamedTuple{(Symbol(C, :_moment_, m.order),)}((missing,))
fresh(m::Moment{C}, ::NamedTuple) where {C} =
    MomentState{C,Symbol(C, :_moment_, m.order),
                Symbol(C, :_sumpower_, m.order)}()
fresh(st::MomentState) = st
@inline update!(::MomentState, row) = nothing
# The value type comes from the dependencies' declared field types, not from
# `typeof` of the runtime quotient — a missing-poisoned power sum would
# otherwise collapse the output column's Union{Missing,...} eltype to Missing.
@inline function value(::MomentState{C,N,D}, vals::NamedTuple) where {C,N,D}
    V = Base.promote_op(/, fieldtype(typeof(vals), D),
                        fieldtype(typeof(vals), :count))
    return NamedTuple{(N,),Tuple{V}}((vals[D] / vals.count,))
end

"""
    Mean(column) -> Summarizer

The mean of `column`. Produces the output column `Symbol(column, :_mean)`,
e.g. `Mean(:x)` produces `:x_mean`. The mean of no rows is `missing`.

A dependent summarizer, computed as [`Sum`](@ref)`(column)` divided by
[`Count`](@ref) — those are folded alongside it but appear in the output only
if requested themselves. The output column's element type is the division's
result (`Int` input divides to `Float64`, `Float32` stays `Float32`).
"""
struct Mean{C} <: Summarizer end
Mean(column::Symbol) = Mean{column}()

struct MeanState{C,N,S} <: SummarizerState end

dependencies(::Mean{C}) where {C} = (Count(), Sum(C))
emptyvalue(::Mean{C}) where {C} = NamedTuple{(Symbol(C, :_mean),)}((missing,))
fresh(::Mean{C}, ::NamedTuple) where {C} =
    MeanState{C,Symbol(C, :_mean),Symbol(C, :_sum)}()
fresh(st::MeanState) = st
@inline update!(::MeanState, row) = nothing
@inline function value(::MeanState{C,N,S}, vals::NamedTuple) where {C,N,S}
    V = Base.promote_op(/, fieldtype(typeof(vals), S), fieldtype(typeof(vals), :count))
    return NamedTuple{(N,),Tuple{V}}((vals[S] / vals.count,))
end

"""
    Variance(column; corrected = true) -> Summarizer

The variance of `column`, following `Statistics.var`: divided by `n - 1` when
`corrected` (the default), by `n` otherwise. Produces the output column
`Symbol(column, :_variance)`, e.g. `Variance(:x)` produces `:x_variance`. The
variance of no rows is `missing`; the corrected variance of a single row is
`NaN` (`0.0` when `corrected = false`).

A dependent summarizer, computed from [`Count`](@ref), [`Sum`](@ref)`(column)`,
and [`SumPower`](@ref)`(column, 2)` by the identity
`(Σx² − (Σx)²/n) / (n − corrected)`; those are folded alongside it but appear
in the output only if requested themselves. The output column's element type is
the computation's result (`Float64` for integer input, `Float32` for
`Float32`).

`corrected` is not part of the output name, so a corrected and an uncorrected
`Variance` of the same column cannot be requested together in one call — they
would share `:x_variance` and collapse under the name-keyed deduplication.
"""
struct Variance{C} <: Summarizer
    corrected::Bool
end
Variance(column::Symbol; corrected::Bool = true) = Variance{column}(corrected)

# R (the corrected flag) is baked into the state type so the derived value
# stays fieldless and inferrable; the divisor is `n - Int(R)`.
struct VarianceState{C,N,S,Q,R} <: SummarizerState end

dependencies(::Variance{C}) where {C} = (Count(), Sum(C), SumPower(C, 2))
emptyvalue(::Variance{C}) where {C} =
    NamedTuple{(Symbol(C, :_variance),)}((missing,))
fresh(v::Variance{C}, ::NamedTuple) where {C} =
    VarianceState{C,Symbol(C, :_variance),Symbol(C, :_sum),
                  Symbol(C, :_sumpower_, 2),v.corrected}()
fresh(st::VarianceState) = st
@inline update!(::VarianceState, row) = nothing
@inline function value(::VarianceState{C,N,S,Q,R}, vals::NamedTuple) where {C,N,S,Q,R}
    Sf = fieldtype(typeof(vals), S)
    Qf = fieldtype(typeof(vals), Q)
    V = Base.promote_op(/, Base.promote_op(-, Qf,
            Base.promote_op(/, Base.promote_op(*, Sf, Sf), Int)), Int)
    s = vals[S]
    q = vals[Q]
    n = vals.count
    return NamedTuple{(N,),Tuple{V}}(((q - s * s / n) / (n - Int(R)),))
end

"""
    Std(column; corrected = true) -> Summarizer

The standard deviation of `column`, following `Statistics.std`: the square
root of [`Variance`](@ref)`(column; corrected)`. Produces the output column
`Symbol(column, :_std)`, e.g. `Std(:x)` produces `:x_std`. The standard
deviation of no rows is `missing`, and of a single corrected row is `NaN`.

A dependent summarizer that folds `Variance(column; corrected)` alongside it
(which appears in the output only if requested itself). A round-off-negative
variance is clamped to zero before the square root, so folding never raises a
`DomainError`.
"""
struct Std{C} <: Summarizer
    corrected::Bool
end
Std(column::Symbol; corrected::Bool = true) = Std{column}(corrected)

struct StdState{C,N,V} <: SummarizerState end

_stdsqrt(::Missing) = missing
_stdsqrt(v) = sqrt(max(v, zero(v)))

dependencies(s::Std{C}) where {C} = (Variance(C; corrected = s.corrected),)
emptyvalue(::Std{C}) where {C} = NamedTuple{(Symbol(C, :_std),)}((missing,))
fresh(::Std{C}, ::NamedTuple) where {C} =
    StdState{C,Symbol(C, :_std),Symbol(C, :_variance)}()
fresh(st::StdState) = st
@inline update!(::StdState, row) = nothing
@inline function value(::StdState{C,N,V}, vals::NamedTuple) where {C,N,V}
    T = Base.promote_op(sqrt, fieldtype(typeof(vals), V))
    return NamedTuple{(N,),Tuple{T}}((_stdsqrt(vals[V]),))
end

"""
    Covariance(a, b; corrected = true) -> Summarizer

The covariance of columns `a` and `b`, following `Statistics.cov`: divided by
`n - 1` when `corrected` (the default), by `n` otherwise. Produces the output
column `Symbol(a, :_, b, :_covariance)`, e.g. `Covariance(:x, :y)` produces
`:x_y_covariance`. The covariance of no rows is `missing`; the corrected
covariance of a single row is `NaN`.

A dependent summarizer, computed from [`Count`](@ref), [`Sum`](@ref)`(a)`,
[`Sum`](@ref)`(b)`, and [`DotProduct`](@ref)`(a, b)` by the identity
`(Σ(ab) − ΣaΣb/n) / (n − corrected)`; those are folded alongside it but appear
in the output only if requested themselves. `Covariance(:x, :x; corrected)`
equals `Variance(:x; corrected)`.

Like [`Variance`](@ref), `corrected` is not part of the output name.
"""
struct Covariance{A,B} <: Summarizer
    corrected::Bool
end
Covariance(a::Symbol, b::Symbol; corrected::Bool = true) =
    Covariance{a,b}(corrected)

struct CovarianceState{A,B,N,D,SA,SB,R} <: SummarizerState end

covname(a, b) = Symbol(a, :_, b, :_covariance)
dependencies(::Covariance{A,B}) where {A,B} =
    (Count(), Sum(A), Sum(B), DotProduct(A, B))
emptyvalue(::Covariance{A,B}) where {A,B} =
    NamedTuple{(covname(A, B),)}((missing,))
fresh(c::Covariance{A,B}, ::NamedTuple) where {A,B} =
    CovarianceState{A,B,covname(A, B),dotname(A, B),Symbol(A, :_sum),
                    Symbol(B, :_sum),c.corrected}()
fresh(st::CovarianceState) = st
@inline update!(::CovarianceState, row) = nothing
@inline function value(::CovarianceState{A,B,N,D,SA,SB,R},
                       vals::NamedTuple) where {A,B,N,D,SA,SB,R}
    Df = fieldtype(typeof(vals), D)
    Saf = fieldtype(typeof(vals), SA)
    Sbf = fieldtype(typeof(vals), SB)
    V = Base.promote_op(/, Base.promote_op(-, Df,
            Base.promote_op(/, Base.promote_op(*, Saf, Sbf), Int)), Int)
    d = vals[D]
    sa = vals[SA]
    sb = vals[SB]
    n = vals.count
    return NamedTuple{(N,),Tuple{V}}(((d - sa * sb / n) / (n - Int(R)),))
end

"""
    Correlation(a, b) -> Summarizer

The Pearson correlation of columns `a` and `b`, following `Statistics.cor`:
[`Covariance`](@ref)`(a, b)` divided by the product of the two columns'
[`Std`](@ref)s, clamped to `[-1, 1]`. Produces the output column
`Symbol(a, :_, b, :_correlation)`, e.g. `Correlation(:x, :y)` produces
`:x_y_correlation`. The correlation of no rows is `missing`, and of a single
row is `NaN`.

Unlike [`Covariance`](@ref) and [`Std`](@ref), `Correlation` takes no
`corrected` keyword: the `n - 1` (or `n`) factor cancels between the
covariance and the standard deviations, so the value is the same either way.
It is a dependent summarizer over `Covariance(a, b)`, `Std(a)`, and `Std(b)`;
those are folded alongside it but appear in the output only if requested
themselves.
"""
struct Correlation{A,B} <: Summarizer end
Correlation(a::Symbol, b::Symbol) = Correlation{a,b}()

struct CorrelationState{A,B,N,CV,SA,SB} <: SummarizerState end

_clampcor(::Missing) = missing
_clampcor(x) = clamp(x, -one(x), one(x))

corname(a, b) = Symbol(a, :_, b, :_correlation)
dependencies(::Correlation{A,B}) where {A,B} =
    (Covariance(A, B), Std(A), Std(B))
emptyvalue(::Correlation{A,B}) where {A,B} =
    NamedTuple{(corname(A, B),)}((missing,))
fresh(::Correlation{A,B}, ::NamedTuple) where {A,B} =
    CorrelationState{A,B,corname(A, B),covname(A, B),Symbol(A, :_std),
                     Symbol(B, :_std)}()
fresh(st::CorrelationState) = st
@inline update!(::CorrelationState, row) = nothing
@inline function value(::CorrelationState{A,B,N,CV,SA,SB},
                       vals::NamedTuple) where {A,B,N,CV,SA,SB}
    Cvf = fieldtype(typeof(vals), CV)
    Saf = fieldtype(typeof(vals), SA)
    Sbf = fieldtype(typeof(vals), SB)
    V = Base.promote_op(/, Cvf, Base.promote_op(*, Saf, Sbf))
    return NamedTuple{(N,),Tuple{V}}((_clampcor(vals[CV] / (vals[SA] * vals[SB])),))
end

# Min/Max/First/Last have no identity element, and all four track one value of
# the input column's type, so they share a state. `F` is the singleton type of
# the combiner (min, max, keepfirst, keeplast), recovered as `F.instance`, so
# `update!` specializes per summarizer. The `seen` flag keeps "no rows folded
# in" distinct from a column holding missing or nothing; the value field is
# typed exactly like the input column — no Union{Missing,T} in the folding
# loop — and is left *undefined* until the first row; `seen` guards every read
# of it.
#
# This relies on `value` never being called on a state that has folded no
# rows, which the transforms guarantee: every key group and every cycle folds
# a row before emitting, a keyless summarize with at least one chunk has at
# least one row, and the no-rows case is answered by `emptyvalue` without ever
# building a state.

keepfirst(a, b) = a
keeplast(a, b) = b

mutable struct TrackState{C,N,T,F} <: SummarizerState
    seen::Bool
    val::T
    TrackState{C,N,T,F}() where {C,N,T,F} = new{C,N,T,F}(false)
    TrackState{C,N,T,F}(seen::Bool, val) where {C,N,T,F} = new{C,N,T,F}(seen, val)
end

fresh(::TrackState{C,N,T,F}) where {C,N,T,F} = TrackState{C,N,T,F}()
@inline function update!(st::TrackState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    st.val = st.seen ? F.instance(st.val, v) : v
    st.seen = true
    return nothing
end
value(st::TrackState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))
function widenstate(st::TrackState{C,N,T,F}, intypes::NamedTuple) where {C,N,T,F}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? TrackState{C,N,T2,F}(true, convert(T2, st.val)) :
        TrackState{C,N,T2,F}()
end

"""
    Min(column) -> Summarizer

Tracks the minimum of `column`. Produces the output column
`Symbol(column, :_min)`, e.g. `Min(:x)` produces `:x_min`, with the same
element type as `column`. The minimum of no rows is `missing`.
"""
struct Min{C} <: Summarizer end
Min(column::Symbol) = Min{column}()

emptyvalue(::Min{C}) where {C} = NamedTuple{(Symbol(C, :_min),)}((missing,))
fresh(::Min{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_min),intypes[C],typeof(min)}()

"""
    Max(column) -> Summarizer

Tracks the maximum of `column`. Produces the output column
`Symbol(column, :_max)`, e.g. `Max(:x)` produces `:x_max`, with the same
element type as `column`. The maximum of no rows is `missing`.
"""
struct Max{C} <: Summarizer end
Max(column::Symbol) = Max{column}()

emptyvalue(::Max{C}) where {C} = NamedTuple{(Symbol(C, :_max),)}((missing,))
fresh(::Max{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_max),intypes[C],typeof(max)}()

"""
    First(column) -> Summarizer

Keeps the value of `column` from the first row folded in. Produces the output
column `Symbol(column, :_first)`, e.g. `First(:x)` produces `:x_first`, with
the same element type as `column`. The first of no rows is `missing`.
"""
struct First{C} <: Summarizer end
First(column::Symbol) = First{column}()

emptyvalue(::First{C}) where {C} = NamedTuple{(Symbol(C, :_first),)}((missing,))
fresh(::First{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_first),intypes[C],typeof(keepfirst)}()

"""
    Last(column) -> Summarizer

Keeps the value of `column` from the most recent row folded in. Produces the
output column `Symbol(column, :_last)`, e.g. `Last(:x)` produces `:x_last`,
with the same element type as `column`. The last of no rows is `missing`.
"""
struct Last{C} <: Summarizer end
Last(column::Symbol) = Last{column}()

emptyvalue(::Last{C}) where {C} = NamedTuple{(Symbol(C, :_last),)}((missing,))
fresh(::Last{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_last),intypes[C],typeof(keeplast)}()
