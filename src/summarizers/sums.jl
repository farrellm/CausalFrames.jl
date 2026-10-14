# The sum family (Count, CountDistinct, Sum, SumPower, Product, DotProduct) and
# the accumulator state it shares.

# The element type Base.sum produces over eltype T: small integers widen to
# Int/UInt, everything else keeps its type. Accumulating at this width is
# type-stable and avoids overflowing the input's own type.
sumtype(::Type{T}) where {T} = Base.promote_op(Base.add_sum, T, T)

# The analogous widths for a product (as `Base.prod`) and a dot product
# (`sumtype` of one `a * b` term).
prodtype(::Type{T}) where {T} = Base.promote_op(Base.mul_prod, T, T)
dottype(::Type{Ta}, ::Type{Tb}) where {Ta,Tb} =
    sumtype(Base.promote_op(*, Ta, Tb))

# The sum family (Sum, SumPower, DotProduct) shares one accumulator state over
# a *term functor*, whose type names the family member and its input columns
# and whose fields carry runtime config (SumPower's exponent). Terms are formed
# in the accumulator type A, so a per-row power or product can't overflow the
# input's own type.

struct ColumnTerm{C} end
struct PowerTerm{C}
    power::Int
end
struct PairProductTerm{A,B} end

@inline termvalue(::ColumnTerm{C}, ::Type{A}, row) where {C,A} =
    convert(A, getproperty(row, C))
@inline termvalue(t::PowerTerm{C}, ::Type{A}, row) where {C,A} =
    convert(A, getproperty(row, C))^t.power
@inline termvalue(::PairProductTerm{Ca,Cb}, ::Type{A}, row) where {Ca,Cb,A} =
    convert(A, getproperty(row, Ca)) * convert(A, getproperty(row, Cb))

# Whether this row's term is `missing` (for a pair, either operand). Only
# reached with the sum state's flag M set.
@inline termmissing(::Union{ColumnTerm{C},PowerTerm{C}}, row) where {C} =
    ismissing(getproperty(row, C))
@inline termmissing(::PairProductTerm{Ca,Cb}, row) where {Ca,Cb} =
    ismissing(getproperty(row, Ca)) || ismissing(getproperty(row, Cb))

# The accumulator type a term folds into, from the promoted input types;
# recomputed by widenstate.
acctype(::ColumnTerm{C}, intypes::NamedTuple) where {C} = sumtype(intypes[C])
acctype(t::PowerTerm{C}, intypes::NamedTuple) where {C} =
    powertype(intypes[C], t.power)
acctype(::PairProductTerm{Ca,Cb}, intypes::NamedTuple) where {Ca,Cb} =
    dottype(intypes[Ca], intypes[Cb])

# The sum family's state. N names the output column, A is the accumulator type,
# T the term functor's type (so update! inlines the term) and S the storage
# (`accstorage(A)`).
#
# A `missing` term would absorb a running sum and could not be subtracted, so,
# as with nonfinite floats, a Missing-admitting accumulator (flag M) counts
# missing terms instead: A is the *non-missing* type, only present terms enter
# `acc`, and `value` is `missing` while `missings` is positive. The count
# balances under `downdate!`, so the state stays invertible. With M false every
# test of `missings` folds away.
mutable struct AccumState{N,A,T,M,S} <: SummarizerState
    term::T
    acc::S
    missings::Int
end

# The sum family's constructor, from the accumulator type. An all-Missing
# column (accumulator type `Missing`) has no non-missing type to fold at, so it
# gets a `MissingState` (below).
function accumfresh(term, N::Symbol, ::Type{A}) where {A}
    A === Missing && return MissingState{N,typeof(term)}(term, 0)
    M = Missing <: A
    An = nonmissingtype(A)
    S = accstorage(An)
    return AccumState{N,An,typeof(term),M,S}(term, acczero(S), 0)
end

AccumState{N,A,T,M,S}(term::T) where {N,A,T,M,S} =
    AccumState{N,A,T,M,S}(term, acczero(S), 0)
fresh(st::AccumState{N,A,T,M,S}) where {N,A,T,M,S} = AccumState{N,A,T,M,S}(st.term)
@inline function fresh!(st::AccumState{N,A,T,M,S}) where {N,A,T,M,S}
    st.acc = acczero(S)
    st.missings = 0
    return st
end
@inline function update!(st::AccumState{N,A,T,M}, row) where {N,A,T,M}
    if M && termmissing(st.term, row)
        st.missings += 1
    else
        st.acc = accadd(st.acc, termvalue(st.term, A, row))
    end
    return nothing
end
@inline function downdate!(st::AccumState{N,A,T,M}, row) where {N,A,T,M}
    if M && termmissing(st.term, row)
        st.missings -= 1
    else
        st.acc = accsub(st.acc, termvalue(st.term, A, row))
    end
    return nothing
end
function combine!(dest::AccumState{N,A,T,M,S}, a::AccumState{N,A,T,M,S},
    b::AccumState{N,A,T,M,S}) where {N,A,T,M,S}
    dest.acc = accmerge(a.acc, b.acc)
    M && (dest.missings = a.missings + b.missings)
    return nothing
end
# The value's field type is the static `maybemissing(A, M)`; a runtime `typeof`
# would let a missing window narrow a dependent summarizer's output type.
value(st::AccumState{N,A,T,M}) where {N,A,T,M} =
    NamedTuple{(N,),Tuple{maybemissing(A, M)}}((
        M && st.missings > 0 ? missing : accvalue(st.acc),))
# A later chunk can widen the accumulator into a compensable float or onto the
# missing-counting path.
function widenstate(st::AccumState{N}, intypes::NamedTuple) where {N}
    w = accumfresh(st.term, N, acctype(st.term, intypes))
    typeof(w) === typeof(st) && return st
    w.acc = widenacc(typeof(w.acc), st.acc)
    w.missings = st.missings
    return w
end

# The state of a sum-family, AgeWeightedSum or Product accumulator over an
# all-`missing` input (accumulator type `Missing`). Every term is missing, so it
# only counts rows; a fresh state is still the identity, and the value is always
# `missing`. `recipe` (a sum-family term functor, or the summarizer) rebuilds the
# real state when a later chunk widens the input, which then takes the count, so
# a window over the missing rows still gives `missing` after the widening.
mutable struct MissingState{N,R} <: SummarizerState
    recipe::R
    n::Int
end

fresh(st::MissingState{N,R}) where {N,R} = MissingState{N,R}(st.recipe, 0)
@inline fresh!(st::MissingState) = (st.n = 0; st)
@inline update!(st::MissingState, row) = (st.n += 1; nothing)
@inline downdate!(st::MissingState, row) = (st.n -= 1; nothing)
combine!(dest::MissingState{N,R}, a::MissingState{N,R}, b::MissingState{N,R}) where {N,R} =
    (dest.n = a.n + b.n; nothing)
value(::MissingState{N}) where {N} = NamedTuple{(N,),Tuple{Missing}}((missing,))
function widenstate(st::MissingState{N}, intypes::NamedTuple) where {N}
    w = refresh(st.recipe, N, intypes)
    w isa MissingState && return st
    return withmissings!(w, st.n)
end

# The state a wider input calls for, from a MissingState's recipe.
refresh(term::Union{ColumnTerm,PowerTerm,PairProductTerm}, N::Symbol,
    intypes::NamedTuple) = accumfresh(term, N, acctype(term, intypes))
refresh(s::Summarizer, ::Symbol, intypes::NamedTuple) = fresh(s, intypes)

# Fold `n` missing rows into a fresh state, whose type admits `missing`.
withmissings!(w::AccumState, n::Int) = (w.missings = n; w)

"""
    Count() -> Summarizer

The number of rows, in the column `:count` (`Int`, `0` for no rows). Alongside
`using MLJ`, which exports its own `Count`, write `CausalFrames.Count()`.
"""
struct Count <: GroupSummarizer end

mutable struct CountState <: SummarizerState
    n::Int
end

emptyvalue(::Count) = (; count = 0)
fresh(::Count, ::NamedTuple) = CountState(0)
fresh(::CountState) = CountState(0)
@inline fresh!(st::CountState) = (st.n = 0; st)
@inline update!(st::CountState, row) = (st.n += 1; nothing)
@inline downdate!(st::CountState, row) = (st.n -= 1; nothing)
combine!(dest::CountState, a::CountState, b::CountState) =
    (dest.n = a.n + b.n; nothing)
value(st::CountState) = (; count = st.n)

"""
    CountDistinct(column::ColumnSpec) -> Summarizer

The number of distinct values of `column`, in `:{column}_countdistinct`
(`Int`, `0` for no rows).

`missing` counts as a value, so `1, missing, 1` has two distinct values; filter
out `missing` upstream for SQL's `count(DISTINCT x)`. The state holds every
distinct value seen, so memory is O(distinct values) per summary; in a sliding
window it also counts each value's rows, so it can drop a value when its last
row leaves.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct CountDistinct{C} <: GroupSummarizer end
CountDistinct(column::ColumnSpec) = withterms(CountDistinct{colname(column)}(), column)

mutable struct CountDistinctState{C,N,T} <: SummarizerState
    seen::Set{T}
    CountDistinctState{C,N,T}() where {C,N,T} = new{C,N,T}(Set{T}())
end

emptyvalue(::CountDistinct{C}) where {C} =
    NamedTuple{(Symbol(C, :_countdistinct),)}((0,))
fresh(::CountDistinct{C}, intypes::NamedTuple) where {C} =
    CountDistinctState{C,Symbol(C, :_countdistinct),intypes[C]}()
fresh(::CountDistinctState{C,N,T}) where {C,N,T} = CountDistinctState{C,N,T}()
# `empty!` keeps the set's slots, so repeated zeroing doesn't allocate.
@inline fresh!(st::CountDistinctState) = (empty!(st.seen); st)
@inline update!(st::CountDistinctState{C}, row) where {C} =
    (push!(st.seen, getproperty(row, C)); nothing)
value(st::CountDistinctState{C,N}) where {C,N} =
    NamedTuple{(N,),Tuple{Int}}((length(st.seen),))
function combine!(dest::CountDistinctState{C,N,T}, a::CountDistinctState{C,N,T},
    b::CountDistinctState{C,N,T}) where {C,N,T}
    # `dest` may alias either argument, in which case it can't be cleared.
    if dest === a
        union!(dest.seen, b.seen)
    elseif dest === b
        union!(dest.seen, a.seen)
    else
        empty!(dest.seen)
        union!(dest.seen, a.seen, b.seen)
    end
    return nothing
end
function widenstate(st::CountDistinctState{C,N,T},
    intypes::NamedTuple) where {C,N,T}
    T2 = intypes[C]
    T2 === T && return st
    widened = CountDistinctState{C,N,T2}()
    union!(widened.seen, st.seen)
    return widened
end

# The windowed state: a row count per distinct value, dropping a value when its
# last row leaves. The non-window folds keep the Set, since a Dict count
# increment hashes twice (measured 1.4-1.7x slower; see DESIGN.md).
struct CountDistinctWindowState{C,N,T} <: SummarizerState
    counts::Dict{T,Int}
end

freshwindowed(::CountDistinct{C}, intypes::NamedTuple) where {C} =
    CountDistinctWindowState{C,Symbol(C, :_countdistinct),intypes[C]}(
        Dict{intypes[C],Int}())
fresh(::CountDistinctWindowState{C,N,T}) where {C,N,T} =
    CountDistinctWindowState{C,N,T}(Dict{T,Int}())
@inline fresh!(st::CountDistinctWindowState) = (empty!(st.counts); st)
@inline function update!(st::CountDistinctWindowState{C}, row) where {C}
    v = getproperty(row, C)
    st.counts[v] = get(st.counts, v, 0) + 1
    return nothing
end
@inline function downdate!(st::CountDistinctWindowState{C}, row) where {C}
    v = getproperty(row, C)
    c = st.counts[v]
    c == 1 ? delete!(st.counts, v) : (st.counts[v] = c - 1)
    return nothing
end
value(st::CountDistinctWindowState{C,N}) where {C,N} =
    NamedTuple{(N,),Tuple{Int}}((length(st.counts),))

"""
    Sum(column::ColumnSpec) -> Summarizer

The sum of `column`, in `:{column}_sum` (`0` for no rows). The element type is
what `Base.sum` gives: small integers widen (`Int32` to `Int64`), other types
are kept (`Float32` stays `Float32`). Floats use compensated summation, with
NaN and ±Inf counted separately so a rolling window recovers once they leave.

# Arguments
- `column`: a column name, or a row term `name => f` summing `f(row)` (see
  [`ColumnSpec`](@ref)).

```jldoctest
df = DataFrame(time = 1:3, high = [5.0, 6.0, 8.0], low = [4.0, 4.5, 5.0])
p = readtable(df) |> summarize(Sum(:range => r -> r.high - r.low))
DataFrame(load(Context(0, 10), p))

# output

1×2 DataFrame
 Row │ time   range_sum
     │ Int64  Float64
─────┼──────────────────
   1 │    10        5.5
```
"""
struct Sum{C} <: GroupSummarizer end
Sum(column::ColumnSpec) = withterms(Sum{colname(column)}(), column)

emptyvalue(::Sum{C}) where {C} = NamedTuple{(Symbol(C, :_sum),)}((0,))
fresh(::Sum{C}, intypes::NamedTuple) where {C} =
    accumfresh(ColumnTerm{C}(), Symbol(C, :_sum), sumtype(intypes[C]))

"""
    SumPower(column::ColumnSpec, n::Integer) -> Summarizer

The sum of `column ^ n`, in `:{column}_sumpower_{n}` (`0` for no rows). The
element type follows [`Sum`](@ref) applied to `column ^ n`; each term is raised
in that widened type, so it cannot overflow the input type. `SumPower(:x, 1)`
is a separate column from `Sum(:x)`.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
- `n`: the exponent, part of the output name.
"""
struct SumPower{C} <: GroupSummarizer
    power::Int
end
SumPower(column::ColumnSpec, power::Integer) =
    withterms(SumPower{colname(column)}(Int(power)), column)

powertype(::Type{T}, ::Int) where {T} = sumtype(Base.promote_op(^, T, Int))

emptyvalue(s::SumPower{C}) where {C} =
    NamedTuple{(Symbol(C, :_sumpower_, s.power),)}((0,))

# PowerTerm's exponent is a *field*, so `^` can't specialize on it and every
# row pays a runtime power. The two common exponents (1 from `Moment(c, 1)`, 2
# from every variance, covariance, correlation and regression) use terms whose
# exponent is in the *type* instead: ColumnTerm (`x`) and PairProductTerm{C,C}
# (`x * x`), about 3x faster per row. The output name and accumulator type are
# unchanged (`powertype(T, 1) === sumtype(T)`, `powertype(T, 2) ===
# dottype(T, T)`).
#
# Values are bit-identical to `x^n` except at `n = 2` over floats whose square
# lands near underflow, where `x * x` is correctly rounded and `^` can be 1 ULP
# off (CPU-dependent). No NaN, ±Inf, signed-zero or subnormal classification
# differs, which is what the compensated states rely on. (On Julia 1.10,
# `(-0.0)^1` is `0.0`, a `^` bug; the accumulator starts at +0.0 either way.)
# See notes/sumpower-terms.md; these properties are tested.
function powerterm(::Type{Val{C}}, power::Int) where {C}
    power == 1 && return ColumnTerm{C}()
    power == 2 && return PairProductTerm{C,C}()
    return PowerTerm{C}(power)
end

# The term is raised to the power in the accumulator's type, then classified:
# NaN^0 and Inf^0 are the finite term 1.0, as in `sum(x .^ 0)`.
fresh(s::SumPower{C}, intypes::NamedTuple) where {C} =
    accumfresh(powerterm(Val{C}, s.power), Symbol(C, :_sumpower_, s.power),
        powertype(intypes[C], s.power))

# A monoid but not a group: dividing a row back out fails at zero and
# truncates for integers.
"""
    Product(column::ColumnSpec) -> Summarizer

The product of `column`, in `:{column}_product` (`1` for no rows). The element
type is what `Base.prod` gives: small integers widen (`Int32` to `Int64`),
other types are kept.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Product{C} <: MonoidSummarizer end
Product(column::ColumnSpec) = withterms(Product{colname(column)}(), column)

mutable struct ProductState{C,N,A} <: SummarizerState
    total::A
end

emptyvalue(::Product{C}) where {C} = NamedTuple{(Symbol(C, :_product),)}((1,))
# An all-Missing column has no `1` to start from, so it gets a `MissingState`.
function fresh(::Product{C}, intypes::NamedTuple) where {C}
    A = prodtype(intypes[C])
    A === Missing && return MissingState{Symbol(C, :_product),Product{C}}(Product{C}(), 0)
    return ProductState{C,Symbol(C, :_product),A}(convert(A, 1))
end
fresh(::ProductState{C,N,A}) where {C,N,A} = ProductState{C,N,A}(convert(A, 1))
@inline fresh!(st::ProductState{C,N,A}) where {C,N,A} =
    (st.total = convert(A, 1); st)
@inline update!(st::ProductState{C}, row) where {C} =
    (st.total *= getproperty(row, C); nothing)
combine!(dest::ProductState{C,N,A}, a::ProductState{C,N,A},
    b::ProductState{C,N,A}) where {C,N,A} =
    (dest.total = a.total * b.total; nothing)
value(st::ProductState{C,N,A}) where {C,N,A} = NamedTuple{(N,),Tuple{A}}((st.total,))
function widenstate(st::ProductState{C,N,A}, intypes::NamedTuple) where {C,N,A}
    A2 = prodtype(intypes[C])
    A2 === A && return st
    return ProductState{C,N,A2}(convert(A2, st.total))
end
withmissings!(w::ProductState, n::Int) = (n > 0 && (w.total = missing); w)

# Emits, under its own name N, the value a dependency computed under D, which
# the topological expansion puts in `vals` first. Fieldless, so it joins
# DerivedState below. It lets a summarizer symmetric in two columns fold under
# one canonical argument order while answering to the name the caller asked
# for (see DESIGN.md's "Symmetric summarizers").
struct AliasState{N,D} <: SummarizerState end

# The value type comes from the dependency's declared field type, not `typeof`
# of the value, so a missing value keeps the column's Union{Missing,...} eltype.
@inline function value(::AliasState{N,D}, vals::NamedTuple) where {N,D}
    V = fieldtype(typeof(vals), D)
    return NamedTuple{(N,),Tuple{V}}((vals[D],))
end

"""
    DotProduct(a::ColumnSpec, b::ColumnSpec) -> Summarizer

The sum of `a * b`, in `:{a}_{b}_dotproduct` (`0` for no rows). The element type
widens as for [`Sum`](@ref), and terms are formed in that type. `DotProduct(:y,
:x)` shares the accumulator of `DotProduct(:x, :y)` (and of any
[`Covariance`](@ref) or [`LinearRegression`](@ref) needing it).

# Arguments
- `a`, `b`: column names or row terms `name => f` (see [`ColumnSpec`](@ref)).
  A row term is named, and so ordered, by its `name`.

```jldoctest
df = DataFrame(time = 1:2, high = [6.0, 9.0], low = [4.0, 6.0],
    volume = [10, 20])
tp = r -> (r.high + r.low) / 2
p = readtable(df) |> summarize(DotProduct(:tp => tp, :volume))
DataFrame(load(Context(0, 10), p))

# output

1×2 DataFrame
 Row │ time   tp_volume_dotproduct
     │ Int64  Float64
─────┼─────────────────────────────
   1 │    10                 200.0
```
"""
struct DotProduct{A,B} <: GroupSummarizer end
DotProduct(a::ColumnSpec, b::ColumnSpec) =
    withterms(DotProduct{colname(a),colname(b)}(), a, b)

dotname(a, b) = Symbol(a, :_, b, :_dotproduct)

# Σab and Σba are the same number, so every summarizer needing one asks for it
# in sorted argument order and they share the accumulator (DESIGN.md,
# "Symmetric summarizers").
canonicalpair(a::Symbol, b::Symbol) = isless(b, a) ? (b, a) : (a, b)
canonicaldot(a::Symbol, b::Symbol) = DotProduct(canonicalpair(a, b)...)
canonicaldotname(a::Symbol, b::Symbol) = dotname(canonicalpair(a, b)...)

emptyvalue(::DotProduct{A,B}) where {A,B} = NamedTuple{(dotname(A, B),)}((0,))
# Reversed arguments name the canonical accumulator as their one dependency and
# rename its value; the sorted form is the accumulator itself.
dependencies(::DotProduct{A,B}) where {A,B} =
    isless(B, A) ? (DotProduct(B, A),) : ()
# The classified term is the per-row product, so Inf * 0.0 counts as NaN.
fresh(::DotProduct{A,B}, intypes::NamedTuple) where {A,B} =
    isless(B, A) ? AliasState{dotname(A, B),dotname(B, A)}() :
    accumfresh(PairProductTerm{A,B}(), dotname(A, B),
        dottype(intypes[A], intypes[B]))
