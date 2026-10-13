"""
    WindowValues(column::ColumnSpec) -> Summarizer

The values of `column` in the order they were folded: the accumulator
[`MeanAbsDev`](@ref) reads, for a summarizer of your own to depend on too when
it needs every value but not their order. It is not exported; write
`CausalFrames.WindowValues`.

Its value, in `:{column}_windowvalues` (`missing` for no rows), is its state
itself, *borrowed*: read it within the same emission and never keep it. Don't
request it as an output column, as every row would share the one state. The
state has these fields to read:

- `vals` and `head`: the values that are neither `missing` nor NaN are
  `vals[head:end]`, oldest first, in a `Vector` of `column`'s non-missing
  element type;
- `missings`, `nans`: how many `missing` and NaN values are folded in.

Memory is O(rows) per summary, so O(window) in a sliding window. Adding a row
appends and removing one advances `head`, each O(1) amortized: removal relies
on [`downdate!`](@ref)'s oldest-first law. Where the order is needed, use
[`CausalFrames.SortedValues`](@ref CausalFrames.SortedValues) instead.

# Arguments
- `column`: the column to collect, or a row term `name => f` (see [`ColumnSpec`](@ref)).

```jldoctest
st = CausalFrames.fresh(CausalFrames.WindowValues(:x), (; x = Float64))
foreach(v -> CausalFrames.update!(st, (; x = v)), [3.0, NaN, 1.0, 2.0])
wv = CausalFrames.value(st).x_windowvalues
(wv.vals[wv.head:end], wv.nans)

# output

([3.0, 1.0, 2.0], 1)
```
"""
struct WindowValues{C} <: GroupSummarizer end
WindowValues(column::ColumnSpec) = withterms(WindowValues{colname(column)}(), column)

# A Vector with a moving head rather than a ring of fixed capacity: an
# ordinary state grows without bound under `addsummarycolumns`, and a window's
# state is sized by its rows, so one layout serves both. `downdate!` advances
# `head` and compacts once the dead prefix is at least half the vector, so both
# directions are O(1) amortized and a steady window never allocates.
# `missing` and NaN are counted, not stored, as in SortedState; M flags a
# Missing-admitting column. `scratch` is the buffer `combine!` swaps with `vals`.
mutable struct WindowValuesState{C,N,T,M} <: SummarizerState
    vals::Vector{T}
    scratch::Vector{T}
    head::Int
    nans::Int
    missings::Int
end

WindowValuesState{C,N,T,M}() where {C,N,T,M} = WindowValuesState{C,N,T,M}(T[], T[], 1, 0, 0)

windowvaluesname(C::Symbol) = Symbol(C, :_windowvalues)

function windowvaluesfresh(::Val{C}, ::Type{A}) where {C,A}
    M = Missing <: A
    return WindowValuesState{C,windowvaluesname(C),nonmissingtype(A),M}()
end

emptyvalue(::WindowValues{C}) where {C} =
    NamedTuple{(windowvaluesname(C),)}((missing,))
fresh(::WindowValues{C}, intypes::NamedTuple) where {C} =
    windowvaluesfresh(Val(C), intypes[C])
fresh(::WindowValuesState{C,N,T,M}) where {C,N,T,M} = WindowValuesState{C,N,T,M}()
@inline function fresh!(st::WindowValuesState)
    empty!(st.vals)
    st.head = 1
    st.nans = 0
    st.missings = 0
    return st
end
@inline function update!(st::WindowValuesState{C}, row) where {C}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings += 1
    elseif isnanvalue(v)
        st.nans += 1
    else
        push!(st.vals, v)
    end
    return nothing
end
# The row leaving is the oldest folded (the oldest-first law), so it is
# `vals[head]` unless it was a counted `missing` or NaN.
@inline function downdate!(st::WindowValuesState{C}, row) where {C}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings -= 1
    elseif isnanvalue(v)
        st.nans -= 1
    else
        h = st.head += 1
        vals = st.vals
        if h > 32 && 2 * (h - 1) >= length(vals)
            n = length(vals) - h + 1
            copyto!(vals, 1, vals, h, n)
            resize!(vals, n)
            st.head = 1
        end
    end
    return nothing
end
# `a`'s rows precede `b`'s. Built in `dest`'s scratch vector and swapped in
# after both inputs are read, so `dest` may alias `a` or `b`.
function combine!(dest::WindowValuesState{C,N,T,M}, a::WindowValuesState{C,N,T,M},
    b::WindowValuesState{C,N,T,M}) where {C,N,T,M}
    na = length(a.vals) - a.head + 1
    nb = length(b.vals) - b.head + 1
    out = resize!(dest.scratch, na + nb)
    copyto!(out, 1, a.vals, a.head, na)
    copyto!(out, na + 1, b.vals, b.head, nb)
    nans = a.nans + b.nans
    missings = a.missings + b.missings
    dest.scratch = dest.vals
    dest.vals = out
    dest.head = 1
    dest.nans, dest.missings = nans, missings
    return nothing
end
value(st::WindowValuesState{C,N,T,M}) where {C,N,T,M} =
    NamedTuple{(N,),Tuple{WindowValuesState{C,N,T,M}}}((st,))
function widenstate(st::WindowValuesState{C,N,T,M},
    intypes::NamedTuple) where {C,N,T,M}
    w = windowvaluesfresh(Val(C), intypes[C])
    typeof(w) === typeof(st) && return st
    append!(w.vals, view(st.vals, st.head:length(st.vals)))
    w.nans, w.missings = st.nans, st.missings
    return w
end

"""
    MeanAbsDev(column::ColumnSpec) -> Summarizer

The mean absolute deviation of `column` about its mean, `Σ|x - mean| / n`, in
`:{column}_meanabsdev` (`missing` for no rows). This is Excel's `AVEDEV`,
TA-Lib's `AVGDEV` and the deviation in its `CCI`. Integer input gives `Float64`,
and `Float32` stays `Float32`. A `missing` in the rows gives `missing` and a NaN
gives NaN, and in a sliding window it recovers once that row leaves. It is
computed from [`CausalFrames.WindowValues`](@ref CausalFrames.WindowValues) and
[`Mean`](@ref), so memory is O(window) per key, and each emission scans the
window's values once.

# Arguments
- `column`: the column to summarize, or a row term `name => f` (see [`ColumnSpec`](@ref)).

```jldoctest
df = DataFrame(time = 1:5, x = [1, 2, 6, 3, 3])
p = readtable(df) |> addrollingcolumns((b3 = Bars(3),), MeanAbsDev(:x))
DataFrame(load(Context(0, 10), p))

# output

5×3 DataFrame
 Row │ time   x      b3_x_meanabsdev
     │ Int64  Int64  Float64?
─────┼───────────────────────────────
   1 │     1      1    missing
   2 │     2      2    missing
   3 │     3      6          2.0
   4 │     4      3          1.55556
   5 │     5      3          1.33333
```
"""
struct MeanAbsDev{C} <: GroupSummarizer end
MeanAbsDev(column::ColumnSpec) = withterms(MeanAbsDev{colname(column)}(), column)

struct MeanAbsDevState{N,D,M} <: SummarizerState end

dependencies(::MeanAbsDev{C}) where {C} = (WindowValues(C), Mean(C))
emptyvalue(::MeanAbsDev{C}) where {C} =
    NamedTuple{(Symbol(C, :_meanabsdev),)}((missing,))
fresh(::MeanAbsDev{C}, ::NamedTuple) where {C} =
    MeanAbsDevState{Symbol(C, :_meanabsdev),windowvaluesname(C),Symbol(C, :_mean)}()
@inline value(::MeanAbsDevState{N,D,M}, vals::NamedTuple) where {N,D,M} =
    meanabsdev(Val(N), vals[D], vals[M], fieldtype(typeof(vals), M))

# The window's values hold every finite row (NaN and missing are only counted),
# so the scan is the whole window; their order is not needed, which is why this
# reads WindowValues rather than SortedValues.
@inline function meanabsdev(::Val{N}, st::WindowValuesState{C,S,T,M}, mean,
    ::Type{A}) where {N,C,S,T,M,A}
    V0 = absdevtype(T, nonmissingtype(A))
    V = M ? Union{Missing,V0} : V0
    M && st.missings > 0 && return NamedTuple{(N,),Tuple{V}}((missing,))
    st.nans > 0 && return NamedTuple{(N,),Tuple{V}}((nanof(V0),))
    # mean is present here, but the compiler can't know it; the test splits
    # the Union
    ismissing(mean) && return NamedTuple{(N,),Tuple{V}}((missing,))
    vs = view(st.vals, st.head:length(st.vals))
    return NamedTuple{(N,),Tuple{V}}((absdevmean(vs, mean),))
end

# An `@simd` loop rather than `sum(f, v)`: on Julia 1.10, `sum` over a closure
# reaches a runtime-dispatched empty-collection fallback, though `v` is never
# empty here. `@simd` lets the additions reassociate, so the loop vectorizes as
# `sum` does; without it the loop is ten times slower at a 1,000-row window.
# When 1.10 support is dropped, consider going back to `sum(x -> abs(x - m), v)`:
# re-check JET and the allocation tests, and benchmark it against this loop
# (`@simd` measured slightly faster than `sum` at windows of 14 to 1,000 rows).
@inline function absdevmean(v::AbstractVector, m)
    s = zero(abs(zero(eltype(v)) - m))
    @simd for i in eachindex(v)
        @inbounds s += abs(v[i] - m)
    end
    return s / length(v)
end

# The scan's type from the value and mean types, so `value` needs no inference
# of the scan itself. An all-missing column (T = Union{}) has no scan to type,
# so its output is Missing.
absdevtype(::Type{T}, ::Type{F}) where {T,F} = typeof(abs(zero(T) - zero(F)) / 1)
absdevtype(::Type{Union{}}, ::Type{F}) where {F} = Union{}
