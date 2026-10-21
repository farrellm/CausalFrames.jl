# The order statistics: the sorted-values accumulator and the Quantile, Median
# and PercentRank dependents over it.

"""
    SortedValues(column::ColumnSpec) -> Summarizer

The values of `column` in sorted order: the accumulator [`Quantile`](@ref),
[`Median`](@ref) and [`PercentRank`](@ref) read, for a summarizer of your own to
depend on too. It is not exported; write `CausalFrames.SortedValues`.

Its value, in `:{column}_sortedvalues` (`missing` for no rows), is its state
itself, *borrowed*: read it within the same emission and never keep it. Don't
request it as an output column, as every row would share the one state. The
state has three fields to read:

- `vals`: the values that are neither `missing` nor NaN, sorted by `isless`, as
  a `Vector` of `column`'s non-missing element type;
- `missings`, `nans`: how many `missing` and NaN values are folded in.

Memory is O(rows) per summary, so O(window) in a sliding window, where adding
and removing a row each cost a binary search and a shift of up to the window's
values.

# Arguments
- `column`: the column to collect, or a row term `name => f` (see [`ColumnSpec`](@ref)).
  Its values must be ordered by `isless`.

```jldoctest
st = CausalFrames.fresh(CausalFrames.SortedValues(:x), (; x = Float64))
foreach(v -> CausalFrames.update!(st, (; x = v)), [3.0, NaN, 1.0, 2.0])
sv = CausalFrames.value(st).x_sortedvalues
(sv.vals, sv.nans)

# output

([1.0, 2.0, 3.0], 1)
```
"""
struct SortedValues{C} <: GroupSummarizer end
SortedValues(column::ColumnSpec) = withterms(SortedValues{colname(column)}(), column)

# A sorted Vector rather than a tree or skip list: a binary search plus a
# memmove stays cheap far past typical window sizes (DESIGN.md has the
# measurements). `missing` and NaN are counted, not stored, so the vector is
# totally ordered by `isless` and both recover once their row leaves. M flags a
# Missing-admitting column. `scratch` is the buffer `combine!` swaps with `vals`.
mutable struct SortedState{C,N,T,M} <: SummarizerState
    vals::Vector{T}
    scratch::Vector{T}
    nans::Int
    missings::Int
end

SortedState{C,N,T,M}() where {C,N,T,M} = SortedState{C,N,T,M}(T[], T[], 0, 0)

sortedname(C::Symbol) = Symbol(C, :_sortedvalues)

# An all-`missing` column gets T = Union{}: nothing is stored, and the
# dependents' output type is Missing.
function sortedfresh(::Val{C}, ::Type{A}) where {C,A}
    M = Missing <: A
    return SortedState{C,sortedname(C),nonmissingtype(A),M}()
end

@inline isnanvalue(x) = false
@inline isnanvalue(x::AbstractFloat) = isnan(x)

emptyvalue(::SortedValues{C}) where {C} = NamedTuple{(sortedname(C),)}((missing,))
fresh(::SortedValues{C}, intypes::NamedTuple) where {C} =
    sortedfresh(Val(C), intypes[C])
fresh(::S) where {S<:SortedState} = S()
@inline function fresh!(st::SortedState)
    empty!(st.vals)
    st.nans = 0
    st.missings = 0
    return st
end
# A new value goes after its equals, keeping fold order; removal may take any
# equal copy, since equal under `isless` is `isequal`.
@inline function update!(st::SortedState{C}, row) where {C}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings += 1
    elseif isnanvalue(v)
        st.nans += 1
    else
        vals = st.vals
        insert!(vals, searchsortedlast(vals, v) + 1, v)
    end
    return nothing
end
@inline function downdate!(st::SortedState{C}, row) where {C}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings -= 1
    elseif isnanvalue(v)
        st.nans -= 1
    else
        vals = st.vals
        deleteat!(vals, searchsortedfirst(vals, v))
    end
    return nothing
end
# Merged into `dest`'s scratch vector and swapped in after both inputs are
# read, so `dest` may alias `a` or `b`.
function combine!(dest::SortedState{C,N,T,M}, a::SortedState{C,N,T,M},
    b::SortedState{C,N,T,M}) where {C,N,T,M}
    out = resize!(dest.scratch, length(a.vals) + length(b.vals))
    mergesorted!(out, a.vals, b.vals)
    nans = a.nans + b.nans
    missings = a.missings + b.missings
    dest.scratch = dest.vals
    dest.vals = out
    dest.nans, dest.missings = nans, missings
    return nothing
end
value(st::SortedState{C,N,T,M}) where {C,N,T,M} =
    NamedTuple{(N,),Tuple{SortedState{C,N,T,M}}}((st,))
# Widening keeps the order, since every schema promotion is monotone.
function widenstate(st::SortedState{C,N,T,M}, intypes::NamedTuple) where {C,N,T,M}
    w = sortedfresh(Val(C), intypes[C])
    typeof(w) === typeof(st) && return st
    append!(w.vals, st.vals)
    w.nans, w.missings = st.nans, st.missings
    return w
end

# A stable merge of two sorted vectors into `out`, `a`'s values first on ties.
function mergesorted!(out::Vector, a::Vector, b::Vector)
    i, j, k = 1, 1, 1
    na, nb = length(a), length(b)
    @inbounds while i <= na && j <= nb
        if isless(b[j], a[i])
            out[k] = b[j]
            j += 1
        else
            out[k] = a[i]
            i += 1
        end
        k += 1
    end
    @inbounds while i <= na
        out[k] = a[i]
        i += 1
        k += 1
    end
    @inbounds while j <= nb
        out[k] = b[j]
        j += 1
        k += 1
    end
    return out
end

"""
    Quantile(column::ColumnSpec, p; interpolation = :linear) -> Summarizer

The `p` quantiles of `column`, one column per probability, in
`:{column}_quantile_{100p}` (`:x_quantile_50` for `0.5`, `:x_quantile_2_5` for
`0.025`; `missing` for no rows). A `missing` in the rows gives `missing` and a
NaN gives NaN; in a sliding window the quantiles recover once that row leaves.
Computed from [`CausalFrames.SortedValues`](@ref CausalFrames.SortedValues), so memory is
O(window) per key.

# Arguments
- `column`: the column to summarize, or a row term `name => f` (see [`ColumnSpec`](@ref)).
  Its values must be ordered by `isless`,
  and `:linear` must be able to interpolate them.
- `p`: a probability in `[0, 1]`, or a non-empty collection of distinct ones,
  giving one column each in the order given. Any other value is an
  `ArgumentError`.

# Keywords
- `interpolation = :linear`: how to pick the quantile from the sorted values.
  `:linear` interpolates between them as `Statistics.quantile` does by default
  (Hyndman and Fan's type 7, up to rounding), so integer input gives `Float64`. `:nearestrank`
  takes the `k`-th smallest value, for the smallest `k` with `k/n ≥ p` (type 1,
  and TA-Lib's `PERCENTILE`), keeping `column`'s element type. It compares `k/n`
  and `p` in floating point, so `p = 0.07` over 100 rows gives the 7th value.
  Any other symbol is an `ArgumentError`. It is not part of the output name, so
  both forms of one `p` cannot be requested together.

```jldoctest
df = DataFrame(time = 1:5, x = [4, 1, 3, 5, 2])
p = readtable(df) |>
    addrollingcolumns((w3 = 3,), Quantile(:x, [0.25, 0.5]))
DataFrame(load(Context(0, 10), p))

# output

5×4 DataFrame
 Row │ time   x      w3_x_quantile_25  w3_x_quantile_50
     │ Int64  Int64  Float64?          Float64?
─────┼──────────────────────────────────────────────────
   1 │     1      4              4.0                4.0
   2 │     2      1              1.75               2.5
   3 │     3      3              2.0                3.0
   4 │     4      5              2.5                3.5
   5 │     5      2              1.75               2.5
```
"""
struct Quantile{C,PS,I} <: GroupSummarizer end

const INTERPOLATIONS = (:linear, :nearestrank)

function Quantile(column::ColumnSpec, ps; interpolation::Symbol = :linear)
    interpolation in INTERPOLATIONS || throw(
        ArgumentError(
            "Quantile interpolation must be :linear or :nearestrank, got " *
            repr(interpolation)),
    )
    qs = Tuple(Float64(p) for p in ps)
    isempty(qs) && throw(ArgumentError("Quantile requires at least one probability"))
    for p in qs
        0 <= p <= 1 || throw(ArgumentError(
            "Quantile probability must be in [0, 1], got $p"))
    end
    allunique(qs) || throw(ArgumentError(
        "Quantile probabilities must be unique, got $qs"))
    # probabilities closer than the name's twelve digits would share a name
    c = colname(column)
    outs = quantilenames(c, qs)
    allunique(outs) || throw(ArgumentError(
        "Quantile output columns must be unique, got $outs"))
    return withterms(Quantile{c,qs,interpolation}(), column)
end
Quantile(column::ColumnSpec, p::Real; kwargs...) = Quantile(column, (p,); kwargs...)

# The percentage to twelve significant digits (so 0.07 names `7`, though
# 100 * 0.07 is 7.000000000000001), with `_` for the decimal point.
function quantilesuffix(p::Float64)
    r = round(100 * p; sigdigits = 12)
    isinteger(r) && return string(Int(r))
    return replace(string(r), '.' => '_')
end
quantilenames(C::Symbol, PS::Tuple) =
    map(p -> Symbol(C, :_quantile_, quantilesuffix(p)), PS)

"""
    Median(column::ColumnSpec) -> Summarizer

The median of `column`, as `Statistics.median` up to rounding, in `:{column}_median`
(`missing` for no rows): [`Quantile`](@ref)`(column, 0.5)` under its own name,
sharing its accumulator, so integer input gives `Float64`. A `missing` in the
rows gives `missing` and a NaN gives NaN.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

```jldoctest
df = DataFrame(time = 1:4, x = [4, 1, 3, 5])
DataFrame(load(Context(0, 10), readtable(df) |> summarize(Median(:x))))

# output

1×2 DataFrame
 Row │ time   x_median
     │ Int64  Float64
─────┼─────────────────
   1 │    10       3.5
```
"""
struct Median{C} <: GroupSummarizer end
Median(column::ColumnSpec) = withterms(Median{colname(column)}(), column)

# Fieldless: the output names NS, the accumulator's name D, the probabilities
# PS and the interpolation I are type parameters, so `value` infers and the map
# over PS unrolls. Median is a QuantileState of one probability.
struct QuantileState{NS,D,PS,I} <: DerivedState end

dependencies(::Quantile{C}) where {C} = (SortedValues(C),)
emptyvalue(::Quantile{C,PS}) where {C,PS} =
    NamedTuple{quantilenames(C, PS)}(map(_ -> missing, PS))
fresh(::Quantile{C,PS,I}, ::NamedTuple) where {C,PS,I} =
    QuantileState{quantilenames(C, PS),sortedname(C),PS,I}()

dependencies(::Median{C}) where {C} = (SortedValues(C),)
emptyvalue(::Median{C}) where {C} = NamedTuple{(Symbol(C, :_median),)}((missing,))
fresh(m::Median{C}, ::NamedTuple) where {C} =
    QuantileState{(outname(m),),sortedname(C),(0.5,),:linear}()

@inline value(::QuantileState{NS,D,PS,I}, vals::NamedTuple) where {NS,D,PS,I} =
    quantilevalue(Val(NS), vals[D], Val(PS), Val(I))

# `st`'s type is the declared field type, so M and T are the schema's.
@inline function quantilevalue(::Val{NS}, st::SortedState{C,N,T,M}, ::Val{PS},
    ::Val{I}) where {NS,C,N,T,M,PS,I}
    V0 = I === :linear ? Base.promote_op(linearquantile, Vector{T}, Float64) : T
    V = maybemissing(V0, M)
    K = length(PS)
    M && st.missings > 0 &&
        return NamedTuple{NS,NTuple{K,V}}(ntuple(_ -> missing, Val(K)))
    st.nans > 0 &&
        return NamedTuple{NS,NTuple{K,V}}(ntuple(_ -> nanof(V0), Val(K)))
    vs = st.vals
    return NamedTuple{NS,NTuple{K,V}}(map(p -> orderstat(vs, p, Val(I)), PS))
end

# Only a float column counts NaNs, so the fallback is unreachable.
@inline nanof(::Type{V}) where {V<:AbstractFloat} = V(NaN)
@noinline nanof(::Type{V}) where {V} =
    throw(ArgumentError("Quantile: a NaN was counted in a column without one"))

@inline orderstat(v::Vector, p::Float64, ::Val{:linear}) = linearquantile(v, p)
@inline orderstat(v::Vector, p::Float64, ::Val{:nearestrank}) =
    @inbounds v[nearestrank(length(v), p)]

# `Statistics.quantile`'s type 7 (alpha = beta = 1) over sorted `v`, written out
# because `quantile(v, p; sorted = true)` scans all of `v` on every call. This
# is Statistics 1.11.5's arithmetic, whose `fma` places an exactly integral
# position exactly; earlier versions can differ in the last bit.
@inline function linearquantile(v::Vector, p::Float64)
    n = length(v)
    aleph = fma(n, p, 1.0 - p)
    j = clamp(trunc(Int, aleph), 1, n - 1)
    γ = clamp(aleph - j, 0, 1)
    if n == 1
        a = @inbounds v[1]
        b = a
    else
        a = @inbounds v[j]
        b = @inbounds v[j+1]
    end
    # when a ≉ b, b - a may overflow; when a ≈ b, the weighted form may not
    # increase with γ
    if isfinite(a) && isfinite(b) && (!(a isa Number) || !(b isa Number) || a ≈ b)
        return a + γ * (b - a)
    else
        return (1 - γ) * a + γ * b
    end
end

# The smallest k with k/n ≥ p, compared in floating point. `ceil(p * n)` can be
# one off where p * n rounds across an integer (0.07 * 100); the correctly
# rounded k/n equals p whenever the exact ratio is p's decimal, matching
# TA-Lib's integer `ceil(P * n / 100)` for a percentage P.
@inline function nearestrank(n::Int, p::Float64)
    k = clamp(ceil(Int, p * n), 1, n)
    while k > 1 && (k - 1) / n >= p
        k -= 1
    end
    while k < n && k / n < p
        k += 1
    end
    return k
end

"""
    PercentRank(column::ColumnSpec) -> Summarizer

The fraction of the other rows whose `column` is strictly below the newest
row's, in `:{column}_percentrank` (`Float64`, `missing` for no rows): for `n`
rows, `count(< newest) / (n - 1)`, so `0` for a strict minimum, `1` for a strict
maximum and `NaN` for a single row. This is Excel's `PERCENTRANK.INC` of the
newest value, and TA-Lib's `PERCENTRANK` over a period of `n - 1` divided by 100.
A `missing` in the rows gives `missing` and a NaN gives NaN; in a sliding window
it recovers once that row leaves. Computed from
[`CausalFrames.SortedValues`](@ref CausalFrames.SortedValues) and [`Last`](@ref), so memory
is O(window) per key.

# Arguments
- `column`: the column to rank, or a row term `name => f` (see [`ColumnSpec`](@ref)).
  Its values must be ordered by `isless`.

```jldoctest
df = DataFrame(time = 1:5, x = [4, 1, 3, 5, 3])
p = readtable(df) |> addrollingcolumns((w4 = 4,), PercentRank(:x))
DataFrame(load(Context(0, 10), p))

# output

5×3 DataFrame
 Row │ time   x      w4_x_percentrank
     │ Int64  Int64  Float64?
─────┼────────────────────────────────
   1 │     1      4            NaN
   2 │     2      1              0.0
   3 │     3      3              0.5
   4 │     4      5              1.0
   5 │     5      3              0.25
```
"""
struct PercentRank{C} <: GroupSummarizer end
PercentRank(column::ColumnSpec) = withterms(PercentRank{colname(column)}(), column)

struct PercentRankState{N,D,L} <: DerivedState end

dependencies(::PercentRank{C}) where {C} = (SortedValues(C), Last(C))
emptyvalue(::PercentRank{C}) where {C} =
    NamedTuple{(Symbol(C, :_percentrank),)}((missing,))
fresh(r::PercentRank{C}, ::NamedTuple) where {C} =
    PercentRankState{outname(r),sortedname(C),outname(Last(C))}()
@inline value(::PercentRankState{N,D,L}, vals::NamedTuple) where {N,D,L} =
    percentrank(Val(N), vals[D], vals[L])

@inline function percentrank(::Val{N}, st::SortedState{C,D,T,M},
    newest) where {N,C,D,T,M}
    V = maybemissing(Float64, M)
    M && st.missings > 0 && return NamedTuple{(N,),Tuple{V}}((missing,))
    st.nans > 0 && return NamedTuple{(N,),Tuple{V}}((NaN,))
    # newest is present here, but the compiler can't know it; the test splits
    # the Union
    vs = st.vals
    below = ismissing(newest) ? 0 : searchsortedfirst(vs, newest) - 1
    return NamedTuple{(N,),Tuple{V}}((below / (length(vs) - 1),))
end
