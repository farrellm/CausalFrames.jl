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

Abstract supertype of summarizers: immutable configurations (typically a column
name, held as a type parameter so the output names are known to the compiler)
that the summarizing transforms fold over rows. A summarizer implements:

- [`emptyvalue`](@ref)`(s)`: the summary of no rows, a `NamedTuple` keyed by
  output column name;
- [`fresh`](@ref)`(s, intypes)`: a zero [`SummarizerState`](@ref) for input
  columns with the element types `intypes`;
- optionally [`dependencies`](@ref)`(s)`, with the two-argument
  [`value`](@ref), to compute its value from other summarizers'.

Summarizers with the same output names are treated as the same summarizer and
share state. Dependencies are folded but appear in the output only if
requested. Subtype [`MonoidSummarizer`](@ref) or [`GroupSummarizer`](@ref) to
unlock faster window algorithms.
"""
abstract type Summarizer end

"""
    MonoidSummarizer <: Summarizer

A summarizer whose states [`combine!`](@ref) associatively, with a
[`fresh`](@ref) state as identity. Window transforms answer each window from a
segment tree of partial states — O(log window) — instead of re-folding it.
"""
abstract type MonoidSummarizer <: Summarizer end

"""
    GroupSummarizer <: MonoidSummarizer

A monoid summarizer whose windowed state ([`freshwindowed`](@ref)) can remove
its oldest row with [`downdate!`](@ref). Window transforms slide such a state
in O(1) amortized per row, removing rows as they leave the window, unless
[`isinvertible`](@ref) says the state can no longer be inverted.
"""
abstract type GroupSummarizer <: MonoidSummarizer end

"""
    SummarizerState

Abstract supertype of the running state of one summarization, built by
[`fresh`](@ref)`(s, intypes)`. Its value fields are concretely typed, so the
output columns are too. A state implements:

- [`fresh`](@ref)`(st)`: a zero state of the same type;
- [`update!`](@ref)`(st, row)`: fold in one row;
- [`value`](@ref)`(st)`: the current summary, a `NamedTuple` keyed by output
  column name;
- optionally [`fresh!`](@ref)`(st)`, [`widenstate`](@ref)`(st, intypes)`, and,
  for structured summarizers, [`combine!`](@ref), [`downdate!`](@ref) and
  [`isinvertible`](@ref).

A [`GroupSummarizer`](@ref) may also build a separate windowed state with
[`freshwindowed`](@ref).
"""
abstract type SummarizerState end

"""
    emptyvalue(s::Summarizer) -> NamedTuple

The summary of no rows, keyed by output column name. Transforms read output
names from it before any data arrives, and use it when there is no data to
build a state from.
"""
function emptyvalue end

"""
    fresh(s::Summarizer, intypes::NamedTuple) -> SummarizerState
    fresh(st::SummarizerState) -> SummarizerState

A zero state. The first form builds it from `s` and the input column element
types `intypes`, read as [`update!`](@ref) reads the row (`intypes[column]` for
`row[column]`). The second returns a new zero state of the same concrete type as
`st`, used for each key group and cycle. Transforms never mutate the
summarizers they are given.
"""
function fresh end

"""
    freshwindowed(s::Summarizer, intypes::NamedTuple) -> SummarizerState

A zero state for a sliding window, which the window transforms
[`update!`](@ref) as rows arrive and [`downdate!`](@ref) as they leave, oldest
first. Defaults to [`fresh`](@ref)`(s, intypes)`. A
[`GroupSummarizer`](@ref) whose ordinary state cannot remove rows implements it
instead: the windowed `First` keeps every value in the window, where the
ordinary one keeps a single value.

The windowed state needs `fresh`, `fresh!`, `update!`, `downdate!`, `value` and
optionally `isinvertible`, and its `value` must have the ordinary state's type.
It is never combined or widened.
"""
freshwindowed(s::Summarizer, intypes::NamedTuple) = fresh(s, intypes)

"""
    fresh!(st::SummarizerState) -> SummarizerState

Zero `st` in place, where possible, and return it; callers use the returned
value. The default returns `fresh(st)`. Implementing it avoids an allocation per
state per cycle, window query or row in the window transforms. A state whose
value is only read after a row has been folded (`Min`, `First`, …) may keep a
stale value.
"""
fresh!(st::SummarizerState) = fresh(st)

"""
    update!(st::SummarizerState, row)

Fold one row into `st`. `row` supports `row.name` and `row[:name]`, including
`row.time`.
"""
function update! end

"""
    value(st::SummarizerState) -> NamedTuple
    value(st::SummarizerState, vals::NamedTuple) -> NamedTuple

The current summary, keyed by output column name; its value types are the
output column element types. Only called on a state that has folded at least
one row (the summary of no rows is [`emptyvalue`](@ref)).

The two-argument form also receives `vals`, the values of every summarizer
earlier in dependency order, including all of [`dependencies`](@ref)`(s)`. It
defaults to the one-argument form; a dependent summarizer implements it instead.
"""
function value end

value(st::SummarizerState, ::NamedTuple) = value(st)

"""
    widenstate(st::SummarizerState, intypes::NamedTuple) -> SummarizerState

`st` rebuilt for the wider input element types `intypes`, keeping its
accumulated value. Called when a later chunk widens a column's type (say `Int`
to `Float64`). Defaults to returning `st`, which suits states whose type does
not depend on the input.
"""
widenstate(st::SummarizerState, ::NamedTuple) = st

"""
    dependencies(s::Summarizer) -> Tuple

The summarizers whose values `s` reads in the two-argument [`value`](@ref).
They are expanded recursively and deduplicated by output name, so a dependency
also requested by the user is folded once, and they appear in the output only
if requested. Defaults to `()`.
"""
dependencies(::Summarizer) = ()

"""
    combine!(dest::SummarizerState, a::SummarizerState, b::SummarizerState)

Set `dest` to the state folding `a`'s rows then `b`'s would give. Required for
a [`MonoidSummarizer`](@ref): it must be associative, with a [`fresh`](@ref)
state as identity on both sides. Every row of `a` precedes every row of `b` in
stream order, so order-sensitive states (`First`, `Last`) can combine. All three
states have the same type and `dest` may alias `a` or `b`, so read the inputs
before writing.
"""
function combine! end

"""
    downdate!(st::SummarizerState, row)

Remove `row`, the oldest row still folded into `st`, inverting its
[`update!`](@ref). Callers remove rows in the order they folded them, so a state
may rely on that: [`AgeWeightedSum`](@ref) and the windowed `Min`, `First`
and `Last` do. Required of a [`GroupSummarizer`](@ref)'s windowed state
([`freshwindowed`](@ref)). The built-in sums are exact for integers; for floats
they use compensated summation and count NaN, ±Inf and `missing` terms
separately, so those remove exactly and finite terms leave only small round-off.
"""
function downdate! end

"""
    isinvertible(st::SummarizerState) -> Bool

Whether [`downdate!`](@ref) inverts [`update!`](@ref) for this windowed
state's actual accumulator type. Defaults to `true`; a state that folds an
absorbing value it cannot recover from returns `false`, and window transforms
then fold that summarizer through a segment tree instead.
"""
isinvertible(::SummarizerState) = true

# Fold one row into — or, for group states, back out of — every state in a
# tuple. Over a concrete state tuple the foreach unrolls and each update!
# dispatches statically, so the shared spelling costs nothing.
@inline updateall!(states::Tuple, row) = foreach(st -> update!(st, row), states)
@inline downdateall!(states::Tuple, row) =
    foreach(st -> downdate!(st, row), states)

# Zero a whole state tuple in place, returning the tuple to use — the same
# must-use-the-result contract as `fresh!` itself, since a state that cannot be
# zeroed in place returns a new object. Over a concrete tuple the map unrolls
# and each `fresh!` dispatches statically, so for the built-in (mutable) states
# this is pure field writes and allocates nothing.
@inline freshall!(states::Tuple) = map(fresh!, states)

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

# Floating-point sum accumulators use Kahan-Babuška-Neumaier compensated
# summation, and nonfinite terms are counted rather than folded in: only
# finite terms enter the (total, comp) pair, while NaN and signed-infinity
# terms bump Int counts, and `compvalue` reconstructs the IEEE result
# `Base.sum` would produce (any NaN, or infinities of both signs, -> NaN; one
# infinity sign -> that infinity). Keeping nonfinites out of the running pair
# is what lets `downdate!` invert exactly once such a row leaves a rolling
# window — folded in naively, NaN absorbs and an evicted infinity leaves
# Inf - Inf = NaN behind. `Missing`-admitting columns count their missing terms
# the same way (the sum state's flag M, over the non-missing type), so they too
# stay invertible. BigFloat is excluded because compensation buys nothing at
# arbitrary precision and a non-isbits Compensated{BigFloat} would heap-allocate
# on every row.
compensable(::Type{T}) where {T} = T <: AbstractFloat && T !== BigFloat

struct Compensated{A<:AbstractFloat}
    total::A     # Neumaier sum of the finite terms only
    comp::A      # running compensation
    nans::Int    # NaN terms currently folded
    posinf::Int  # +Inf terms
    neginf::Int  # -Inf terms
end

compzero(::Type{A}) where {A<:AbstractFloat} =
    Compensated{A}(zero(A), zero(A), 0, 0, 0)

# One Neumaier step. The isfinite guard keeps comp finite when the total
# overflows to infinity from all-finite terms, so compvalue reconstructs the
# Inf that Base.sum would produce rather than Inf + (-Inf) = NaN.
@inline function neumaier(s::A, c::A, x::A) where {A<:AbstractFloat}
    t = s + x
    if isfinite(t)
        c += abs(s) >= abs(x) ? (s - t) + x : (x - t) + s
    end
    return t, c
end

@inline function compadd(a::Compensated{A}, x::A) where {A}
    if isfinite(x)
        t, c = neumaier(a.total, a.comp, x)
        return Compensated{A}(t, c, a.nans, a.posinf, a.neginf)
    elseif isnan(x)
        return Compensated{A}(a.total, a.comp, a.nans + 1, a.posinf, a.neginf)
    elseif x > zero(A)
        return Compensated{A}(a.total, a.comp, a.nans, a.posinf + 1, a.neginf)
    else
        return Compensated{A}(a.total, a.comp, a.nans, a.posinf, a.neginf + 1)
    end
end

# Inverse of compadd over the identically computed term, so the counts always
# balance; negating a finite float is exact.
@inline function compsub(a::Compensated{A}, x::A) where {A}
    if isfinite(x)
        t, c = neumaier(a.total, a.comp, -x)
        return Compensated{A}(t, c, a.nans, a.posinf, a.neginf)
    elseif isnan(x)
        return Compensated{A}(a.total, a.comp, a.nans - 1, a.posinf, a.neginf)
    elseif x > zero(A)
        return Compensated{A}(a.total, a.comp, a.nans, a.posinf - 1, a.neginf)
    else
        return Compensated{A}(a.total, a.comp, a.nans, a.posinf, a.neginf - 1)
    end
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

# The sum family (Sum, SumPower, DotProduct) shares one accumulator state
# (plain or compensated storage) over a *term functor* — the TrackState
# combiner-in-type-parameter idiom applied to the folded quantity. The
# functor's type identifies the family and its input columns; its fields
# carry runtime config (SumPower's exponent). Terms are formed in the
# accumulator's type A, so a per-row power or product cannot overflow the
# way computing it in the input columns' own types would; for the
# compensated (float) states the input column already is A, so the
# conversion is exact either way.

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

# Whether a term is `missing` for this row — a missing input column, or (for a
# pair) either operand missing. Only reached with the sum state's flag M set,
# i.e. when a source column admits Missing; over a non-missing column the
# `ismissing` folds to a compile-time `false`.
@inline termmissing(::ColumnTerm{C}, row) where {C} = ismissing(getproperty(row, C))
@inline termmissing(::PowerTerm{C}, row) where {C} = ismissing(getproperty(row, C))
@inline termmissing(::PairProductTerm{Ca,Cb}, row) where {Ca,Cb} =
    ismissing(getproperty(row, Ca)) || ismissing(getproperty(row, Cb))

# The accumulator type a term folds into, from the (promoted) input column
# types — recomputed by widenstate whenever the schema promotion moves.
acctype(::ColumnTerm{C}, intypes::NamedTuple) where {C} = sumtype(intypes[C])
acctype(t::PowerTerm{C}, intypes::NamedTuple) where {C} =
    powertype(intypes[C], t.power)
acctype(::PairProductTerm{Ca,Cb}, intypes::NamedTuple) where {Ca,Cb} =
    dottype(intypes[Ca], intypes[Cb])

# The storage an accumulator of type A folds into: a `Compensated{A}` for a
# compensable float, A itself otherwise. Each operation dispatches on it, so
# the one state below compiles to the plain or the compensated fold.
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

# Carry an accumulator into a wider storage. Integer and plain-float totals are
# finite as far as the counters know, which is all a plain storage tracked; a
# compensated one keeps its running pair and counters.
widenacc(::Type{S2}, acc) where {S2} = convert(S2, accvalue(acc))
widenacc(::Type{Compensated{A2}}, acc) where {A2} =
    compadd(compzero(A2), convert(A2, acc))
widenacc(::Type{Compensated{A2}}, acc::Compensated) where {A2} =
    widencomp(A2, acc)

@inline maybemissing(::Type{A}, M::Bool) where {A} = M ? Union{Missing,A} : A

# The sum family's one state. N names the output column and A is the realized
# accumulator type, as on every other state; T is the term functor's type, so
# update! inlines the term computation statically; S is the storage
# (`accstorage(A)`).
#
# A `missing` input term absorbs a running sum and cannot be subtracted back
# out, which would force a rolling window off the O(1) running path onto the
# tree. So, exactly as the compensated storage counts nonfinite floats, a
# Missing-admitting accumulator (the flag M, `AgeSumState`'s idiom) counts its
# missing terms instead of folding them in: the accumulation lives at the
# *non-missing* type A (flat — no Union in the hot field), only present terms
# enter it, `missings` tracks the rest, and `value` reports `missing` whenever
# that count is positive. The count balances under `downdate!`, so the state
# stays invertible and a missing row recovers once it leaves the window. With
# M false, `missings` stays zero and every test of it folds away.
mutable struct AccumState{N,A,T,M,S} <: SummarizerState
    term::T
    acc::S
    missings::Int
end

# Shared constructor behind the sum family's fresh methods, from the
# accumulator type. (The `Union{}` guard keeps a pathological all-Missing
# column off the counting path.)
function accumfresh(term, N::Symbol, ::Type{A}) where {A}
    M = Missing <: A && nonmissingtype(A) !== Union{}
    An = M ? nonmissingtype(A) : A
    S = accstorage(An)
    return AccumState{N,An,typeof(term),M,S}(term, acczero(S), 0)
end

fresh(st::AccumState{N,A,T,M,S}) where {N,A,T,M,S} =
    AccumState{N,A,T,M,S}(st.term, acczero(S), 0)
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
# The value's field type is the static `maybemissing(A, M)` — a runtime `typeof`
# would let a missing window collapse a dependent summarizer's output type.
value(st::AccumState{N,A,T,M}) where {N,A,T,M} =
    NamedTuple{(N,),Tuple{maybemissing(A, M)}}((
        M && st.missings > 0 ? missing : accvalue(st.acc),))
# A later chunk can widen the accumulator — into a compensable float, or into a
# Missing-admitting type, which moves it onto the counting path with no missing
# folded in yet.
function widenstate(st::AccumState{N}, intypes::NamedTuple) where {N}
    w = accumfresh(st.term, N, acctype(st.term, intypes))
    typeof(w) === typeof(st) && return st
    w.acc = widenacc(typeof(w.acc), st.acc)
    w.missings = st.missings
    return w
end

# Reinterpret a Compensated at a wider float type, carrying its running pair
# and its nonfinite counters unchanged.
widencomp(::Type{A2}, a::Compensated) where {A2} =
    Compensated{A2}(convert(A2, a.total), convert(A2, a.comp),
        a.nans, a.posinf, a.neginf)

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
    CountDistinct(column::Symbol) -> Summarizer

The number of distinct values of `column`, in `:{column}_countdistinct`
(`Int`, `0` for no rows).

`missing` counts as a value, so `1, missing, 1` has two distinct values; filter
out `missing` upstream for SQL's `count(DISTINCT x)`. The state holds every
distinct value seen, so memory is O(distinct values) per summary; in a sliding
window it also counts each value's rows, so it can drop a value when its last
row leaves.
"""
struct CountDistinct{C} <: GroupSummarizer end
CountDistinct(column::Symbol) = CountDistinct{column}()

mutable struct CountDistinctState{C,N,T} <: SummarizerState
    seen::Set{T}
    CountDistinctState{C,N,T}() where {C,N,T} = new{C,N,T}(Set{T}())
end

emptyvalue(::CountDistinct{C}) where {C} =
    NamedTuple{(Symbol(C, :_countdistinct),)}((0,))
fresh(::CountDistinct{C}, intypes::NamedTuple) where {C} =
    CountDistinctState{C,Symbol(C, :_countdistinct),intypes[C]}()
fresh(::CountDistinctState{C,N,T}) where {C,N,T} = CountDistinctState{C,N,T}()
# `empty!` keeps the set's slots, so the per-cycle, per-interval and per-window
# zeroing is allocation-free once the first fold has sized it.
@inline fresh!(st::CountDistinctState) = (empty!(st.seen); st)
@inline update!(st::CountDistinctState{C}, row) where {C} =
    (push!(st.seen, getproperty(row, C)); nothing)
value(st::CountDistinctState{C,N}) where {C,N} =
    NamedTuple{(N,),Tuple{Int}}((length(st.seen),))
function combine!(dest::CountDistinctState{C,N,T}, a::CountDistinctState{C,N,T},
    b::CountDistinctState{C,N,T}) where {C,N,T}
    # `dest` may alias either argument, so it can only be cleared once both
    # have been read — which, for a union, means not clearing it at all.
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

# The windowed state: a count of rows per distinct value, so removing the
# oldest row drops its value exactly when no other row in the window holds it.
# Kept apart from the Set above because incrementing a count through the public
# Dict API hashes twice per row, which measured 1.4-1.7x slower than `push!` on
# the folds that never remove a row (see DESIGN.md's CountDistinct paragraph).
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
    Sum(column::Symbol) -> Summarizer

The sum of `column`, in `:{column}_sum` (`0` for no rows). The element type is
what `Base.sum` gives: small integers widen (`Int32` to `Int64`), other types
are kept (`Float32` stays `Float32`). Floats use compensated summation, with
NaN and ±Inf counted separately so a rolling window recovers once they leave.
"""
struct Sum{C} <: GroupSummarizer end
Sum(column::Symbol) = Sum{column}()

emptyvalue(::Sum{C}) where {C} = NamedTuple{(Symbol(C, :_sum),)}((0,))
fresh(::Sum{C}, intypes::NamedTuple) where {C} =
    accumfresh(ColumnTerm{C}(), Symbol(C, :_sum), sumtype(intypes[C]))

"""
    SumPower(column::Symbol, n::Integer) -> Summarizer

The sum of `column ^ n`, in `:{column}_sumpower_{n}` (`0` for no rows). The
element type follows [`Sum`](@ref) applied to `column ^ n`; each term is raised
in that widened type, so it cannot overflow the input type. `SumPower(:x, 1)`
is a separate column from `Sum(:x)`.
"""
struct SumPower{C} <: GroupSummarizer
    power::Int
end
SumPower(column::Symbol, power::Integer) = SumPower{column}(Int(power))

powertype(::Type{T}, ::Int) where {T} = sumtype(Base.promote_op(^, T, Int))

emptyvalue(s::SumPower{C}) where {C} =
    NamedTuple{(Symbol(C, :_sumpower_, s.power),)}((0,))

# PowerTerm carries its exponent in a *field*, so `^` cannot specialize on it:
# every row pays a runtime power (`power_by_squaring` for integers, `pow_body`
# for floats) where a move or a single multiply would do. The two exponents that
# actually turn up in this package — 1 from `Moment(c, 1)`, 2 from every
# variance, covariance, correlation and regression — therefore borrow the terms
# whose exponent is in the *type*: ColumnTerm is `x` and PairProductTerm{C,C} is
# `x * x`. Worth about 3x on the per-row fold (see notes/sumpower-terms.md).
#
# The output column keeps its own name and the accumulator type is unchanged
# (`powertype(T, 1) === sumtype(T)` and `powertype(T, 2) === dottype(T, T)` for
# every T the package admits), so nothing about the schema moves.
#
# The *value* is bit-identical for `n = 1`, and for integers and Bool at both
# exponents. At `n = 2` over floats it is not quite: `x * x` is the correctly
# rounded square, while the runtime `^` can be 1 ULP off it for inputs whose
# square lands near underflow (66 of 500k random Float64 bit patterns where it
# was measured; how many depends on the CPU, since `^` rounds its error terms
# differently with and without FMA). The specialization is the more accurate of
# the two there — but it is a change, so it is stated rather than glossed. What the compensated states
# actually require is unaffected: they classify NaN and ±Inf *terms* and carry
# the sign of zero, and no nonfinite or subnormal case differs at either
# exponent. (On Julia 1.10 only, `(-0.0)^1` returns `0.0` — a `^` bug fixed in
# 1.11 — so there ColumnTerm is the *more* correct of the two; the accumulator
# starts at +0.0 and absorbs the difference either way.) See
# notes/sumpower-terms.md; the properties are tested.
function powerterm(::Type{Val{C}}, power::Int) where {C}
    power == 1 && return ColumnTerm{C}()
    power == 2 && return PairProductTerm{C,C}()
    return PowerTerm{C}(power)
end

# The term is formed in the accumulator's type before raising to the power
# (see termvalue), then classified: NaN^0 and Inf^0 are the finite term 1.0,
# exactly as they contribute to `sum(x .^ 0)`.
fresh(s::SumPower{C}, intypes::NamedTuple) where {C} =
    accumfresh(powerterm(Val{C}, s.power), Symbol(C, :_sumpower_, s.power),
        powertype(intypes[C], s.power))

# A monoid but not a group: dividing a row back out fails outright at zero
# (the total is 0 no matter what else was folded) and truncates for integers.
"""
    Product(column::Symbol) -> Summarizer

The product of `column`, in `:{column}_product` (`1` for no rows). The element
type is what `Base.prod` gives: small integers widen (`Int32` to `Int64`),
other types are kept.
"""
struct Product{C} <: MonoidSummarizer end
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

# A summarizer that emits, under its own name N, a value another summarizer
# already computed under the name D. It folds nothing: the summarizer that
# claims it declares that other one as a dependency, and the topological
# expansion guarantees D is in `vals` by the time this runs. Fieldless, so it
# joins DerivedState below and inherits the no-op combine!/downdate!/fresh!.
#
# This is what lets a summarizer whose value is symmetric in two columns fold
# under one canonical argument order while still answering to the name the
# caller asked for — see DESIGN.md's "Symmetric summarizers".
struct AliasState{N,D} <: SummarizerState end

fresh(st::AliasState) = st
@inline update!(::AliasState, row) = nothing
# The value type comes from the dependency's declared field type, never from
# `typeof` of the value, so a missing-poisoned accumulator keeps the output
# column's Union{Missing,...} eltype instead of collapsing it to Missing.
@inline function value(::AliasState{N,D}, vals::NamedTuple) where {N,D}
    V = fieldtype(typeof(vals), D)
    return NamedTuple{(N,),Tuple{V}}((vals[D],))
end

"""
    DotProduct(a::Symbol, b::Symbol) -> Summarizer

The sum of `a * b`, in `:{a}_{b}_dotproduct` (`0` for no rows). The element type
widens as for [`Sum`](@ref), and terms are formed in that type. `DotProduct(:y,
:x)` shares the accumulator of `DotProduct(:x, :y)` (and of any
[`Covariance`](@ref) or [`LinearRegression`](@ref) needing it).
"""
struct DotProduct{A,B} <: GroupSummarizer end
DotProduct(a::Symbol, b::Symbol) = DotProduct{a,b}()

dotname(a, b) = Symbol(a, :_, b, :_dotproduct)

# Σab and Σba are the same number, so every summarizer needing one asks for it
# under the sorted argument order and they all share the accumulator. See
# DESIGN.md's "Symmetric summarizers" for the rule these two implement.
canonicaldot(a::Symbol, b::Symbol) =
    isless(b, a) ? DotProduct(b, a) : DotProduct(a, b)
canonicaldotname(a::Symbol, b::Symbol) =
    isless(b, a) ? dotname(b, a) : dotname(a, b)

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

"""
    AgeWeightedSum(column::Symbol) -> Summarizer

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
- `column`: the column to weight. Age is counted in rows, in stream order, so
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
AgeWeightedSum(column::Symbol) = AgeWeightedSum{column}()

# The state folds three numbers: the row count n, S1 = Σy and S2 = Σk·y. A new
# row ages every row already folded by one, adding S1 to S2 before joining S1
# itself at weight 0. Removing the oldest row — the only row `downdate!` is ever
# handed — takes back its weight n - 1. Combining ages a's rows by b's count. n
# counts every row, missing and nonfinite included, since age is counted in rows.
#
# A `missing` input is counted rather than folded, as the sum family's Optional*
# states do, but through the flag M (the column admits Missing) instead of a
# second pair of state types: `missings` is always there, and its test folds away
# when the column cannot hold `missing`.
mutable struct AgeSumState{N,C,A,M} <: SummarizerState
    n::Int
    s1::A
    s2::A
    missings::Int
end

# The float form keeps S1 as a `Compensated`, so its counters classify each raw
# NaN and ±Inf input once, on entry. S2 keeps only the finite pair — its counters
# stay zero — and takes its nonfinite classification from S1's, minus the newest
# row's (`newest`: 0 finite or missing, 1 NaN, 2 +Inf, 3 -Inf), whose weight is 0.
mutable struct CompensatedAgeSumState{N,C,A<:AbstractFloat,M} <: SummarizerState
    n::Int
    s1::Compensated{A}
    s2::Compensated{A}
    newest::Int8
    missings::Int
end

agesumname(C::Symbol) = Symbol(C, :_ageweightedsum)

emptyvalue(::AgeWeightedSum{C}) where {C} = NamedTuple{(agesumname(C),)}((0,))
fresh(::AgeWeightedSum{C}, intypes::NamedTuple) where {C} =
    agesumfresh(Val(agesumname(C)), Val(C), sumtype(intypes[C]))

# The representation for accumulator type A, as `accumfresh` picks it: the
# non-missing type carries the fold, compensated when it is a fixed-precision
# float.
function agesumfresh(::Val{N}, ::Val{C}, ::Type{A}) where {N,C,A}
    M = Missing <: A && nonmissingtype(A) !== Union{}
    An = M ? nonmissingtype(A) : A
    compensable(An) && return CompensatedAgeSumState{N,C,An,M}(0, compzero(An),
        compzero(An), Int8(0), 0)
    return AgeSumState{N,C,An,M}(0, convert(An, 0), convert(An, 0), 0)
end

fresh(::AgeSumState{N,C,A,M}) where {N,C,A,M} =
    AgeSumState{N,C,A,M}(0, convert(A, 0), convert(A, 0), 0)
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
function widenstate(st::AgeSumState{N,C,A,M}, intypes::NamedTuple) where {N,C,A,M}
    w = agesumfresh(Val(N), Val(C), sumtype(intypes[C]))
    typeof(w) === typeof(st) && return st
    return agesumfrom(w, st.n, st.s1, st.s2, st.missings)
end

# Carry a plain state's numbers into a freshly built wider one. Integer and
# plain-float totals are finite as far as the counters know, which is all a
# plain state ever tracked.
function agesumfrom(w::AgeSumState{N,C,A,M}, n, s1, s2, missings) where {N,C,A,M}
    return AgeSumState{N,C,A,M}(n, convert(A, s1), convert(A, s2), missings)
end
function agesumfrom(w::CompensatedAgeSumState{N,C,A,M}, n, s1, s2,
    missings) where {N,C,A,M}
    return CompensatedAgeSumState{N,C,A,M}(n, compadd(compzero(A), convert(A, s1)),
        compadd(compzero(A), convert(A, s2)), Int8(0), missings)
end

@inline agesumclass(x::AbstractFloat) =
    isfinite(x) ? Int8(0) : isnan(x) ? Int8(1) : x > 0 ? Int8(2) : Int8(3)

# S2's finite pair plus another pair (S1's, or a scaled one): the two Neumaier
# steps `compmerge` takes, leaving S2's unused counters at zero.
@inline function pairadd(a::Compensated{A}, total::A, comp::A) where {A}
    t, c = neumaier(a.total, a.comp, total)
    t, c = neumaier(t, c, comp)
    return Compensated{A}(t, c, 0, 0, 0)
end

fresh(::CompensatedAgeSumState{N,C,A,M}) where {N,C,A,M} =
    CompensatedAgeSumState{N,C,A,M}(0, compzero(A), compzero(A), Int8(0), 0)
@inline function fresh!(st::CompensatedAgeSumState{N,C,A}) where {N,C,A}
    st.n = 0
    st.s1 = compzero(A)
    st.s2 = compzero(A)
    st.newest = Int8(0)
    st.missings = 0
    return st
end
@inline function update!(st::CompensatedAgeSumState{N,C,A}, row) where {N,C,A}
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
    end
    return nothing
end
@inline function downdate!(st::CompensatedAgeSumState{N,C,A}, row) where {N,C,A}
    v = getproperty(row, C)
    if ismissing(v)
        st.missings -= 1
    else
        x = convert(A, v)
        isfinite(x) && (st.s2 = pairadd(st.s2, -(convert(A, st.n - 1) * x), zero(A)))
        st.s1 = compsub(st.s1, x)
    end
    st.n -= 1
    st.n == 0 && (st.newest = Int8(0))
    return nothing
end
function combine!(dest::CompensatedAgeSumState{N,C,A,M},
    a::CompensatedAgeSumState{N,C,A,M},
    b::CompensatedAgeSumState{N,C,A,M}) where {N,C,A,M}
    k = convert(A, b.n)
    s2 = pairadd(pairadd(a.s2, a.s1.total * k, a.s1.comp * k), b.s2.total, b.s2.comp)
    s1 = compmerge(a.s1, b.s1)
    newest = b.n > 0 ? b.newest : a.newest
    n = a.n + b.n
    missings = a.missings + b.missings
    dest.n, dest.s1, dest.s2, dest.newest, dest.missings = n, s1, s2, newest, missings
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
function widenstate(st::CompensatedAgeSumState{N,C,A,M},
    intypes::NamedTuple) where {N,C,A,M}
    w = agesumfresh(Val(N), Val(C), sumtype(intypes[C]))
    typeof(w) === typeof(st) && return st
    return agesumwiden(w, st)
end
function agesumwiden(w::CompensatedAgeSumState{N,C,A2,M2},
    st::CompensatedAgeSumState) where {N,C,A2,M2}
    return CompensatedAgeSumState{N,C,A2,M2}(st.n, widencomp(A2, st.s1),
        widencomp(A2, st.s2), st.newest, st.missings)
end
# To an arbitrary-precision float: the plain state, from the IEEE values.
agesumwiden(w::AgeSumState, st::CompensatedAgeSumState) =
    agesumfrom(w, st.n, compvalue(st.s1), agesumvalue(st), st.missings)

"""
    Moment(column::Symbol, n::Integer) -> Summarizer

The `n`-th raw moment of `column`, the mean of `column ^ n`, in
`:{column}_moment_{n}` (`missing` for no rows). Computed from
[`SumPower`](@ref)`(column, n)` and [`Count`](@ref); integer input gives
`Float64`.
"""
struct Moment{C} <: GroupSummarizer
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
    Mean(column::Symbol) -> Summarizer

The mean of `column`, in `:{column}_mean` (`missing` for no rows). Computed from
[`Sum`](@ref)`(column)` and [`Count`](@ref); integer input gives `Float64`,
`Float32` stays `Float32`.
"""
struct Mean{C} <: GroupSummarizer end
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
    Variance(column::Symbol; corrected = true) -> Summarizer

The variance of `column`, as `Statistics.var`, in `:{column}_variance`
(`missing` for no rows). Computed from [`Count`](@ref), [`Sum`](@ref) and
[`SumPower`](@ref)`(column, 2)`; integer input gives `Float64`.

# Keywords
- `corrected = true`: divide by `n - 1` (a single row gives `NaN`); `false`
  divides by `n`. It is not part of the output name, so both forms of one column
  cannot be requested together.
"""
struct Variance{C} <: GroupSummarizer
    corrected::Bool
end
Variance(column::Symbol; corrected::Bool = true) = Variance{column}(corrected)

# R (the corrected flag) is baked into the state type so the derived value
# stays fieldless and inferrable; the divisor is `n - Int(R)`.
struct VarianceState{C,N,S,Q,R} <: SummarizerState end

# The compile-time value type of the shared (co)variance identity
# `(q - sa * sb / n) / (n - corrected)`, from the dependencies' declared
# field types (a runtime `typeof` would let one missing collapse the type).
_covtype(::Type{Q}, ::Type{Sa}, ::Type{Sb}) where {Q,Sa,Sb} =
    Base.promote_op(/,
        Base.promote_op(-, Q,
            Base.promote_op(/, Base.promote_op(*, Sa, Sb), Int)), Int)

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
    V = _covtype(Qf, Sf, Sf)
    s = vals[S]
    q = vals[Q]
    n = vals.count
    return NamedTuple{(N,),Tuple{V}}(((q - s * s / n) / (n - Int(R)),))
end

"""
    Std(column::Symbol; corrected = true) -> Summarizer

The standard deviation of `column`, as `Statistics.std`, in `:{column}_std`
(`missing` for no rows): the square root of [`Variance`](@ref), with round-off
negatives clamped to zero.

# Keywords
- `corrected = true`: as for [`Variance`](@ref).
"""
struct Std{C} <: GroupSummarizer
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
    Covariance(a::Symbol, b::Symbol; corrected = true) -> Summarizer

The covariance of `a` and `b`, as `Statistics.cov`, in `:{a}_{b}_covariance`
(`missing` for no rows). Computed from [`Count`](@ref), [`Sum`](@ref) of each
column and [`DotProduct`](@ref)`(a, b)`.

# Keywords
- `corrected = true`: as for [`Variance`](@ref), including that it is not part
  of the output name.
"""
struct Covariance{A,B} <: GroupSummarizer
    corrected::Bool
end
Covariance(a::Symbol, b::Symbol; corrected::Bool = true) =
    Covariance{a,b}(corrected)

struct CovarianceState{A,B,N,D,SA,SB,R} <: SummarizerState end

covname(a, b) = Symbol(a, :_, b, :_covariance)
# The covariance is symmetric, so it reads the canonically ordered dot product
# rather than its own argument order: Covariance(:y, :x) still produces
# :y_x_covariance, but shares the one :x_y_dotproduct accumulator.
dependencies(::Covariance{A,B}) where {A,B} =
    (Count(), Sum(A), Sum(B), canonicaldot(A, B))
emptyvalue(::Covariance{A,B}) where {A,B} =
    NamedTuple{(covname(A, B),)}((missing,))
fresh(c::Covariance{A,B}, ::NamedTuple) where {A,B} =
    CovarianceState{A,B,covname(A, B),canonicaldotname(A, B),Symbol(A, :_sum),
        Symbol(B, :_sum),c.corrected}()
fresh(st::CovarianceState) = st
@inline update!(::CovarianceState, row) = nothing
@inline function value(::CovarianceState{A,B,N,D,SA,SB,R},
    vals::NamedTuple) where {A,B,N,D,SA,SB,R}
    Df = fieldtype(typeof(vals), D)
    Saf = fieldtype(typeof(vals), SA)
    Sbf = fieldtype(typeof(vals), SB)
    V = _covtype(Df, Saf, Sbf)
    d = vals[D]
    sa = vals[SA]
    sb = vals[SB]
    n = vals.count
    return NamedTuple{(N,),Tuple{V}}(((d - sa * sb / n) / (n - Int(R)),))
end

"""
    Correlation(a::Symbol, b::Symbol) -> Summarizer

The Pearson correlation of `a` and `b`, as `Statistics.cor`, clamped to
`[-1, 1]`, in `:{a}_{b}_correlation` (`missing` for no rows, `NaN` for one).
Computed from [`Covariance`](@ref) and the two columns' [`Std`](@ref)s; there is
no `corrected` keyword, since the correction cancels.
"""
struct Correlation{A,B} <: GroupSummarizer end
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

"""
    LinearRegression(predictors, response::Symbol; intercept = true,
                     name = nothing) -> Summarizer

An ordinary least squares fit of `response` on `predictors`.

# Arguments
- `predictors`: a column name, or a collection of distinct names (`K` of them).
  None may be named `intercept`.
- `response`: the response column.

# Keywords
- `intercept = true`: fit a constant term. Without it, `r2` is the uncentered
  R².
- `name = nothing`: a prefix for every output column, `{name}_{column}`.
  Needed to request two regressions together, since both would otherwise
  produce `n`, `r2` and `stderr`.

# Output columns

| column | meaning |
|---|---|
| `n` | rows folded (`Int`, never `missing`) |
| `r2` | coefficient of determination |
| `stderr` | residual standard error |
| `intercept_beta`, `intercept_tstat` | the constant term and its t statistic (only with `intercept`) |
| `{p}_beta`, `{p}_tstat` | per predictor `p`, in order |

```jldoctest
df = DataFrame(time = 1:5, x = [1.0, 2.0, 3.0, 4.0, 5.0], z = [0.0, 1.0, 0.0, 1.0, 1.0],
               y = [3.1, 7.9, 7.2, 12.1, 13.8])
p = readtable(df) |> summarize(LinearRegression([:x, :z], :y; name = :m1))
names(load(Context(0, 10), p))

# output

10-element Vector{String}:
 "time"
 "m1_n"
 "m1_r2"
 "m1_stderr"
 "m1_intercept_beta"
 "m1_intercept_tstat"
 "m1_x_beta"
 "m1_x_tstat"
 "m1_z_beta"
 "m1_z_tstat"
```

The statistics share one element type (`Float64` for integer input) and admit
`missing` if an input does. No rows, or a `missing` in any input, gives
`missing` statistics. A rank-deficient fit (collinear predictors, or too few
rows) gives `NaN` rather than an error; with no residual degrees of freedom the
coefficients are exact but `stderr` and the t statistics are `NaN`.

The fit is computed from [`Count`](@ref), [`SumPower`](@ref)`(·, 2)`,
[`DotProduct`](@ref) and (with an intercept) [`Sum`](@ref), so regressions over
overlapping columns, and [`Covariance`](@ref)s or [`Variance`](@ref)s beside
them, share accumulators. With `K ≥ 2` each emitted row allocates a `K × K`
workspace; `K = 1` allocates nothing.
"""
struct LinearRegression{P,Y} <: GroupSummarizer
    intercept::Bool
    name::Union{Nothing,Symbol}
end

function LinearRegression(predictors, response::Symbol;
    intercept::Bool = true, name::Union{Nothing,Symbol} = nothing)
    ps = Tuple(Symbol(p) for p in predictors)
    isempty(ps) &&
        throw(ArgumentError("LinearRegression requires at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "LinearRegression predictors must be unique, got $ps"))
    # A predictor named `intercept` produces the constant term's own pair of
    # columns. Caught here rather than left to the NamedTuple constructor,
    # whose "duplicate field name" says nothing about which summarizer built
    # it. The check is on the whole name set rather than that one case, so it
    # stays honest if the naming scheme grows.
    outs = regnames(ps, name, intercept)
    allunique(outs) || throw(ArgumentError(
        "LinearRegression output columns must be unique, got $outs"))
    return LinearRegression{ps,response}(intercept, name)
end
# A lone name. A string is one name too, never iterated as a collection of
# one-character names.
LinearRegression(predictor::Union{Symbol,AbstractString}, response::Symbol;
    kwargs...) = LinearRegression((Symbol(predictor),), response; kwargs...)

# Fieldless like every other derived state, and every name it reads back is a
# type parameter so the two-argument `value` infers: NN and SN are the output
# names (the row count, then the statistics), AN every accumulator name read —
# deduplicated, since a predictor may be the response — and SP/SY/QY/GN/DN the
# sums, the response's power sum, the packed upper triangle of the cross-product
# matrix, and the predictor-response cross products. I is the intercept flag,
# baked in like Variance's `corrected` so the state stays fieldless.
struct LinearRegressionState{NN,SN,AN,SP,SY,QY,GN,DN,I} <: SummarizerState end

_regname(::Nothing, base::Symbol) = base
_regname(prefix::Symbol, base::Symbol) = Symbol(prefix, :_, base)

# The output names in emission order: the row count, the two model-level
# statistics, the intercept's pair when there is one, then a (beta, tstat) pair
# per predictor in the order given.
function regnames(P::Tuple, name, intercept::Bool)
    ns = Symbol[_regname(name, :n), _regname(name, :r2), _regname(name, :stderr)]
    if intercept
        push!(ns, _regname(name, :intercept_beta))
        push!(ns, _regname(name, :intercept_tstat))
    end
    for p in P
        push!(ns, _regname(name, Symbol(p, :_beta)))
        push!(ns, _regname(name, Symbol(p, :_tstat)))
    end
    return Tuple(ns)
end

# Cross products go through the canonical (sorted) dot product every symmetric
# summarizer shares. A squared term instead goes to SumPower(c, 2) rather than
# DotProduct(c, c): same value, under the name Variance, Std and Correlation
# already depend on.
crossdep(a::Symbol, b::Symbol) =
    a === b ? SumPower(a, 2) : canonicaldot(a, b)
crossname(a::Symbol, b::Symbol) =
    a === b ? Symbol(a, :_sumpower_2) : canonicaldotname(a, b)

# Entry (i, j), i <= j, of a K x K matrix packed row-major over its upper
# triangle — the layout GN is built in.
@inline gramindex(i::Int, j::Int, K::Int) =
    (i - 1) * K - ((i - 1) * (i - 2)) ÷ 2 + (j - i + 1)

function dependencies(r::LinearRegression{P,Y}) where {P,Y}
    ds = Summarizer[Count(), SumPower(Y, 2)]
    for (i, p) in pairs(P)
        push!(ds, SumPower(p, 2))
        push!(ds, crossdep(p, Y))
        for j in (i+1):length(P)
            push!(ds, crossdep(p, P[j]))
        end
    end
    if r.intercept
        push!(ds, Sum(Y))
        for p in P
            push!(ds, Sum(p))
        end
    end
    return Tuple(ds)
end

function emptyvalue(r::LinearRegression{P,Y}) where {P,Y}
    ns = regnames(P, r.name, r.intercept)
    return NamedTuple{ns}((0, ntuple(_ -> missing, length(ns) - 1)...))
end

function fresh(r::LinearRegression{P,Y}, ::NamedTuple) where {P,Y}
    ns = regnames(P, r.name, r.intercept)
    gn = Tuple(crossname(P[i], P[j]) for i in eachindex(P) for j in i:length(P))
    dn = map(p -> crossname(p, Y), P)
    qy = (Symbol(Y, :_sumpower_2),)
    sy = r.intercept ? (Symbol(Y, :_sum),) : ()
    sp = r.intercept ? map(p -> Symbol(p, :_sum), P) : ()
    an = Tuple(unique((qy..., gn..., dn..., sy..., sp...)))
    return LinearRegressionState{first(ns),Base.tail(ns),an,sp,sy,qy,gn,dn,
        r.intercept}()
end
fresh(st::LinearRegressionState) = st
@inline update!(::LinearRegressionState, row) = nothing

# The dependency values this regression reads, and — separately — the tuple
# type they are *declared* to have. The two must not be confused: `typeof` of
# the values is value-dependent (a poisoned accumulator holding `missing`
# reports `Missing`, not `Union{Missing,_}`), so it both collapses the output
# column's element type and, because it is then not a compile-time constant,
# drops the whole emission into runtime dispatch. `Base.promote_op` asks
# inference for the declared type instead, which is constant.
@inline depvalues(vals::NamedTuple, ::Val{NS}) where {NS} =
    values(NamedTuple{NS}(vals))
@inline deptypes(::Type{V}, ::Val{NS}) where {V<:NamedTuple,NS} =
    Base.promote_op(depvalues, V, Val{NS})

# The statistic columns' shared element type: promote those declared types,
# run the centered identity's arithmetic over the result, then force it
# floating point, since a rank-deficient fit reports NaN.
@inline _promotefields(::Type{Tuple{A}}) where {A} = A
@inline _promotefields(::Type{T}) where {T<:Tuple} =
    promote_type(Base.tuple_type_head(T), _promotefields(Base.tuple_type_tail(T)))

_tofloat(x) = float(x)
_tofloat(::Missing) = missing

@inline function _regtype(::Type{T}) where {T<:Tuple}
    A = _promotefields(T)
    Q = Base.promote_op(-, A, Base.promote_op(/, Base.promote_op(*, A, A), Int))
    return Base.promote_op(_tofloat, Base.promote_op(/, Q, Q))
end

@inline _anymissing(::Tuple{}) = false
@inline _anymissing(t::Tuple) = ismissing(first(t)) || _anymissing(Base.tail(t))

# Narrow a dependency value to the compute type. By the time this runs the
# caller has already returned if anything was missing, but the compiler does
# not know that, so the `ismissing` test is what keeps it dispatch-free: it
# splits the Union explicitly and lets `convert` resolve statically. Leaving it
# to `convert` alone only looks fine on a one-element tuple — Julia
# union-splits that — and goes dynamic as soon as a second Union-typed value
# joins it, which is what the K >= 2 cross-product tuple is.
@inline _conv(::Type{Vc}, x) where {Vc} = ismissing(x) ? zero(Vc) : convert(Vc, x)
@inline _convall(::Type{Vc}, ::Tuple{}) where {Vc} = ()
@inline _convall(::Type{Vc}, t::Tuple) where {Vc} =
    (_conv(Vc, first(t)), _convall(Vc, Base.tail(t))...)

# A dependency read back as a scalar; the empty name tuple is the no-intercept
# case, where there is no such dependency and the value is never used.
@inline _regscalar(::Type{Vc}, ::NamedTuple, ::Val{()}) where {Vc} = zero(Vc)
@inline _regscalar(::Type{Vc}, vals::NamedTuple, ::Val{NS}) where {Vc,NS} =
    _conv(Vc, only(depvalues(vals, Val(NS))))

# Simple regression in closed form. This is the common case and the one that
# runs per row under addsummarycolumns and addrollingcolumns, so it stays on
# scalars and never builds the workspace the general path needs. Every quotient
# that can go 0/0 does so in floating point, and every sqrt argument is clamped
# at zero, so a degenerate window yields NaN rather than raising.
@inline function regstats(::Type{Vc}, ::Val{I}, ::Val{M}, n::Vc,
    g::NTuple{G,Vc}, d::NTuple{1,Vc}, s::Tuple{Vararg{Vc}}, sy::Vc,
    qy::Vc) where {Vc,I,M,G}
    nan = Vc(NaN)
    if I
        sxx = g[1] - s[1] * s[1] / n
        sxy = d[1] - s[1] * sy / n
        sst = qy - sy * sy / n
        dof = n - 2
    else
        sxx = g[1]
        sxy = d[1]
        sst = qy
        dof = n - 1
    end
    sxx > zero(Vc) || return ntuple(_ -> nan, Val(M))
    beta = sxy / sxx
    sse = sst - beta * sxy
    r2 = one(Vc) - sse / sst
    positive = dof > zero(Vc)
    sigma2 = positive ? max(sse, zero(Vc)) / dof : nan
    stderr = sqrt(sigma2)
    tbeta = beta / sqrt(sigma2 / sxx)
    if I
        xbar = s[1] / n
        alpha = sy / n - beta * xbar
        talpha = alpha / sqrt(sigma2 * (one(Vc) / n + xbar * xbar / sxx))
        return (r2, stderr, alpha, talpha, beta, tbeta)
    end
    return (r2, stderr, beta, tbeta)
end

# Multiple regression. The cross-product matrix is symmetric positive
# semidefinite, so Cholesky is the factorization; `check = false` turns rank
# deficiency into a flag rather than a PosDefException, and its inverse gives
# both the coefficients' standard errors and the intercept's quadratic form.
function regstats(::Type{Vc}, ::Val{I}, ::Val{M}, n::Vc, g::NTuple{G,Vc},
    d::NTuple{K,Vc}, s::Tuple{Vararg{Vc}}, sy::Vc, qy::Vc) where {Vc,I,M,G,K}
    nan = Vc(NaN)
    rhs = I ? ntuple(i -> d[i] - s[i] * sy / n, Val(K)) : d
    sst = I ? qy - sy * sy / n : qy
    dof = I ? n - K - 1 : n - K
    A = Matrix{Vc}(undef, K, K)
    for i in 1:K, j in i:K
        A[i, j] = I ? g[gramindex(i, j, K)] - s[i] * s[j] / n :
                  g[gramindex(i, j, K)]
    end
    F = cholesky!(Symmetric(A, :U); check = false)
    issuccess(F) || return ntuple(_ -> nan, Val(M))
    b = Vector{Vc}(undef, K)
    copyto!(b, rhs)
    ldiv!(F, b)
    ssr = zero(Vc)
    for i in 1:K
        ssr += b[i] * rhs[i]
    end
    sse = sst - ssr
    r2 = one(Vc) - sse / sst
    sigma2 = dof > zero(Vc) ? max(sse, zero(Vc)) / dof : nan
    stderr = sqrt(sigma2)
    Ci = inv(F)
    coefs = ntuple(Val(2 * K)) do k
        i = (k + 1) >> 1
        isodd(k) ? b[i] : b[i] / sqrt(sigma2 * Ci[i, i])
    end
    if I
        xbar = ntuple(i -> s[i] / n, Val(K))
        alpha = sy / n
        for i in 1:K
            alpha -= b[i] * xbar[i]
        end
        quad = zero(Vc)
        for i in 1:K, j in 1:K
            quad += xbar[i] * Ci[i, j] * xbar[j]
        end
        talpha = alpha / sqrt(sigma2 * (one(Vc) / n + quad))
        return (r2, stderr, alpha, talpha, coefs...)
    end
    return (r2, stderr, coefs...)
end

@inline function value(::LinearRegressionState{NN,SN,AN,SP,SY,QY,GN,DN,I},
    vals::NamedTuple) where {NN,SN,AN,SP,SY,QY,GN,DN,I}
    V = _regtype(deptypes(typeof(vals), Val(AN)))
    M = length(SN)
    nrow = NamedTuple{(NN,),Tuple{Int}}((vals.count,))
    if Missing <: V && _anymissing(depvalues(vals, Val(AN)))
        return merge(nrow,
            NamedTuple{SN,NTuple{M,V}}(ntuple(_ -> missing, Val(M))))
    end
    Vc = nonmissingtype(V)
    stats = regstats(Vc, Val(I), Val(M), convert(Vc, vals.count),
        _convall(Vc, depvalues(vals, Val(GN))),
        _convall(Vc, depvalues(vals, Val(DN))),
        _convall(Vc, depvalues(vals, Val(SP))),
        _regscalar(Vc, vals, Val(SY)), _regscalar(Vc, vals, Val(QY)))
    return merge(nrow, NamedTuple{SN,NTuple{M,V}}(stats))
end

"""
    SortedValues(column::Symbol) -> Summarizer

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
- `column`: the column to collect. Its values must be ordered by `isless`.

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
SortedValues(column::Symbol) = SortedValues{column}()

# A sorted Vector rather than a balanced tree or skip list: the binary search is
# O(log window) and the insert or delete a memmove of the shorter side, which
# stays cheap and cache-friendly far past the windows indicators use (DESIGN.md
# has the measurements). `missing` and NaN are counted rather than stored, as the
# sum family counts them, so the vector is totally ordered by `isless` and
# either one recovers once its row leaves. M flags a Missing-admitting column,
# as on AgeSumState. `scratch` is the merge buffer `combine!` swaps with `vals`.
mutable struct SortedState{C,N,T,M} <: SummarizerState
    vals::Vector{T}
    scratch::Vector{T}
    nans::Int
    missings::Int
end

SortedState{C,N,T,M}() where {C,N,T,M} = SortedState{C,N,T,M}(T[], T[], 0, 0)

sortedname(C::Symbol) = Symbol(C, :_sortedvalues)

# A column of only `missing` gets T = Union{}: nothing is ever stored, and the
# dependents' output type collapses to Missing.
function sortedfresh(::Val{C}, ::Type{A}) where {C,A}
    M = Missing <: A
    return SortedState{C,sortedname(C),nonmissingtype(A),M}()
end

@inline isnanvalue(x) = false
@inline isnanvalue(x::AbstractFloat) = isnan(x)

emptyvalue(::SortedValues{C}) where {C} = NamedTuple{(sortedname(C),)}((missing,))
fresh(::SortedValues{C}, intypes::NamedTuple) where {C} =
    sortedfresh(Val(C), intypes[C])
fresh(::SortedState{C,N,T,M}) where {C,N,T,M} = SortedState{C,N,T,M}()
@inline function fresh!(st::SortedState)
    empty!(st.vals)
    st.nans = 0
    st.missings = 0
    return st
end
# A new value goes after its equals, so equal values keep their fold order; any
# copy may leave, since equal under `isless` is `isequal`.
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
# Merged into `dest`'s scratch vector, which is none of the inputs' `vals`, and
# swapped in once both have been read, so `dest` may alias `a` or `b`.
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
# Widening keeps the order: every promotion the schema makes (Int to Float64,
# a type to its Missing union) is monotone.
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
    Quantile(column::Symbol, p; interpolation = :linear) -> Summarizer

The `p` quantiles of `column`, one column per probability, in
`:{column}_quantile_{100p}` (`:x_quantile_50` for `0.5`, `:x_quantile_2_5` for
`0.025`; `missing` for no rows). A `missing` in the rows gives `missing` and a
NaN gives NaN; in a sliding window the quantiles recover once that row leaves.
Computed from [`CausalFrames.SortedValues`](@ref CausalFrames.SortedValues), so memory is
O(window) per key.

# Arguments
- `column`: the column to summarize. Its values must be ordered by `isless`,
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

function Quantile(column::Symbol, ps; interpolation::Symbol = :linear)
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
    # Two probabilities closer than the name's twelve digits would share a name
    outs = quantilenames(column, qs)
    allunique(outs) || throw(ArgumentError(
        "Quantile output columns must be unique, got $outs"))
    return Quantile{column,qs,interpolation}()
end
Quantile(column::Symbol, p::Real; kwargs...) = Quantile(column, (p,); kwargs...)

# The percentage, to twelve significant digits so that 0.07 names `7` although
# 100 * 0.07 is 7.000000000000001, with `_` for the decimal point.
function quantilesuffix(p::Float64)
    r = round(100 * p; sigdigits = 12)
    isinteger(r) && return string(Int(r))
    return replace(string(r), '.' => '_')
end
quantilenames(C::Symbol, PS::Tuple) =
    map(p -> Symbol(C, :_quantile_, quantilesuffix(p)), PS)

"""
    Median(column::Symbol) -> Summarizer

The median of `column`, as `Statistics.median` up to rounding, in `:{column}_median`
(`missing` for no rows): [`Quantile`](@ref)`(column, 0.5)` under its own name,
sharing its accumulator, so integer input gives `Float64`. A `missing` in the
rows gives `missing` and a NaN gives NaN.

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
Median(column::Symbol) = Median{column}()

# Fieldless like the other derived states: the output names NS, the
# accumulator's name D, the probabilities PS and the interpolation I are all
# type parameters, so the two-argument `value` infers and the map over PS
# unrolls. Median is a QuantileState of one probability.
struct QuantileState{NS,D,PS,I} <: SummarizerState end

dependencies(::Quantile{C}) where {C} = (SortedValues(C),)
emptyvalue(::Quantile{C,PS}) where {C,PS} =
    NamedTuple{quantilenames(C, PS)}(map(_ -> missing, PS))
fresh(::Quantile{C,PS,I}, ::NamedTuple) where {C,PS,I} =
    QuantileState{quantilenames(C, PS),sortedname(C),PS,I}()

dependencies(::Median{C}) where {C} = (SortedValues(C),)
emptyvalue(::Median{C}) where {C} = NamedTuple{(Symbol(C, :_median),)}((missing,))
fresh(::Median{C}, ::NamedTuple) where {C} =
    QuantileState{(Symbol(C, :_median),),sortedname(C),(0.5,),:linear}()

fresh(st::QuantileState) = st
@inline update!(::QuantileState, row) = nothing
@inline value(::QuantileState{NS,D,PS,I}, vals::NamedTuple) where {NS,D,PS,I} =
    quantilevalue(Val(NS), vals[D], Val(PS), Val(I))

# The accumulator's type is its declared field type, never a missing-dependent
# `typeof`, so M and T are what the schema says.
@inline function quantilevalue(::Val{NS}, st::SortedState{C,N,T,M}, ::Val{PS},
    ::Val{I}) where {NS,C,N,T,M,PS,I}
    V0 = I === :linear ? Base.promote_op(linearquantile, Vector{T}, Float64) : T
    V = M ? Union{Missing,V0} : V0
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

# `Statistics.quantile`'s type 7 (alpha = beta = 1) over sorted `v`, written
# out because `quantile(v, p; sorted = true)` still scans all of `v` for NaN and
# `missing` on every call, which would make each emission O(window). This is the
# arithmetic of Statistics 1.11.5, whose `fma` lands a position that is an exact
# integer exactly; earlier versions add `n * p` unfused, and can differ from it
# in the last bit.
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

# The smallest k with k/n ≥ p, compared in floating point: `ceil(p * n)` can
# land one off where p * n rounds across an integer (0.07 * 100), and k/n is
# the correctly rounded quotient, so it equals p whenever the exact ratio is
# p's decimal. That is TA-Lib's exact integer `ceil(P * n / 100)` for a
# percentage P.
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
    PercentRank(column::Symbol) -> Summarizer

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
- `column`: the column to rank. Its values must be ordered by `isless`.

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
PercentRank(column::Symbol) = PercentRank{column}()

struct PercentRankState{N,D,L} <: SummarizerState end

dependencies(::PercentRank{C}) where {C} = (SortedValues(C), Last(C))
emptyvalue(::PercentRank{C}) where {C} =
    NamedTuple{(Symbol(C, :_percentrank),)}((missing,))
fresh(::PercentRank{C}, ::NamedTuple) where {C} =
    PercentRankState{Symbol(C, :_percentrank),sortedname(C),Symbol(C, :_last)}()
fresh(st::PercentRankState) = st
@inline update!(::PercentRankState, row) = nothing
@inline value(::PercentRankState{N,D,L}, vals::NamedTuple) where {N,D,L} =
    percentrank(Val(N), vals[D], vals[L])

@inline function percentrank(::Val{N}, st::SortedState{C,S,T,M},
    newest) where {N,C,S,T,M}
    V = M ? Union{Missing,Float64} : Float64
    M && st.missings > 0 && return NamedTuple{(N,),Tuple{V}}((missing,))
    st.nans > 0 && return NamedTuple{(N,),Tuple{V}}((NaN,))
    # newest is present here, but only a missing window proves it; the test
    # splits the Union for the compiler
    vs = st.vals
    below = ismissing(newest) ? 0 : searchsortedfirst(vs, newest) - 1
    return NamedTuple{(N,),Tuple{V}}((below / (length(vs) - 1),))
end

# The derived states are fieldless — the summary is computed from the
# dependencies' values at emission time — so combining and downdating them is
# a no-op. The window transforms give such a state no tier at all (tiers.jl);
# these methods serve the one case where it keeps one, a set of nothing but
# fieldless states.
const DerivedState = Union{AliasState,MomentState,MeanState,VarianceState,
    StdState,CovarianceState,CorrelationState,LinearRegressionState,
    QuantileState,PercentRankState}
combine!(::DerivedState, ::DerivedState, ::DerivedState) = nothing
@inline downdate!(::DerivedState, row) = nothing
# Fieldless, so already zero — and immutable, so returning `st` is the whole
# implementation (the default would allocate nothing either, but this keeps
# `freshall!` free of a call that inference has to see through).
@inline fresh!(st::DerivedState) = st

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
# Only `seen` is cleared: a field cannot be un-defined, so a stale value may
# survive where `fresh` would leave the field undefined. Both are equally
# unreadable — `seen` guards every read of `val`, and `value` is only ever
# called on a state that has folded a row.
@inline fresh!(st::TrackState) = (st.seen = false; st)
@inline function update!(st::TrackState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    st.val = st.seen ? F.instance(st.val, v) : v
    st.seen = true
    return nothing
end
# Reads both inputs before writing, so it tolerates dest aliasing a or b; the
# ordered-ranges law is what makes keepfirst/keeplast correct here.
function combine!(dest::TrackState{C,N,T,F}, a::TrackState{C,N,T,F},
    b::TrackState{C,N,T,F}) where {C,N,T,F}
    if a.seen && b.seen
        v = F.instance(a.val, b.val)
        dest.val = v
        dest.seen = true
    elseif a.seen
        dest.val = a.val
        dest.seen = true
    elseif b.seen
        dest.val = b.val
        dest.seen = true
    else
        dest.seen = false
    end
    return nothing
end
value(st::TrackState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))
function widenstate(st::TrackState{C,N,T,F}, intypes::NamedTuple) where {C,N,T,F}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? TrackState{C,N,T2,F}(true, convert(T2, st.val)) :
           TrackState{C,N,T2,F}()
end

# The windowed state of Min, Max and First: a monotonic deque of the window's
# candidate values, oldest at the front. A new value `v` first drops every value `b` at the
# back that it makes redundant — those with `F(b, v)` equal to `v`, which cannot
# be the window's answer while `v` is in it — so the front is always the fold of
# the window. That relies on `F` selecting one of its arguments associatively,
# which `min` and `max` do under `isequal` (NaN, ±0.0 and `missing` included),
# as does `keepfirst` (nothing is ever redundant, so the deque is the window).
# `keeplast` would qualify too — everything is redundant, so the deque holds one
# value — but Last has a cheaper state of its own below.
#
# Rows leave oldest first (the `downdate!` law), so each row's sequence number
# is all eviction needs: the front goes when its row does, and a row already
# dropped as redundant is simply not there. Dead front slots are reclaimed once
# they dominate, the rolling buffer's amortized compaction, so a steady window
# neither grows the vectors nor allocates.
mutable struct WindowTrackState{C,N,T,F} <: SummarizerState
    vals::Vector{T}
    seqs::Vector{Int}  # each live value's row number, counted from the last zero
    front::Int         # first live slot
    pushed::Int        # rows folded since the last zero
    popped::Int        # rows removed since the last zero
end

WindowTrackState{C,N,T,F}() where {C,N,T,F} =
    WindowTrackState{C,N,T,F}(T[], Int[], 1, 0, 0)

fresh(::WindowTrackState{C,N,T,F}) where {C,N,T,F} = WindowTrackState{C,N,T,F}()
@inline function fresh!(st::WindowTrackState)
    empty!(st.vals)
    empty!(st.seqs)
    st.front = 1
    st.pushed = 0
    st.popped = 0
    return st
end
@inline function update!(st::WindowTrackState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    vals, seqs = st.vals, st.seqs
    while length(vals) >= st.front && isequal(F.instance(@inbounds(vals[end]), v), v)
        pop!(vals)
        pop!(seqs)
    end
    push!(vals, v)
    push!(seqs, st.pushed += 1)
    return nothing
end
@inline function downdate!(st::WindowTrackState, row)
    st.popped += 1
    seqs = st.seqs
    if st.front <= length(seqs) && @inbounds(seqs[st.front]) == st.popped
        st.front += 1
        dead = st.front - 1
        if dead == length(seqs)
            empty!(st.vals)
            empty!(seqs)
            st.front = 1
        elseif dead >= 32 && 2 * dead >= length(seqs)
            deleteat!(st.vals, 1:dead)
            deleteat!(seqs, 1:dead)
            st.front = 1
        end
    end
    return nothing
end
value(st::WindowTrackState{C,N,T}) where {C,N,T} =
    NamedTuple{(N,),Tuple{T}}((@inbounds(st.vals[st.front]),))

"""
    Min(column::Symbol) -> Summarizer

The minimum of `column`, in `:{column}_min`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the minimum, up to the whole window when `column` rises.
"""
struct Min{C} <: GroupSummarizer end
Min(column::Symbol) = Min{column}()

emptyvalue(::Min{C}) where {C} = NamedTuple{(Symbol(C, :_min),)}((missing,))
fresh(::Min{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_min),intypes[C],typeof(min)}()

"""
    Max(column::Symbol) -> Summarizer

The maximum of `column`, in `:{column}_max`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the maximum, up to the whole window when `column` falls.
"""
struct Max{C} <: GroupSummarizer end
Max(column::Symbol) = Max{column}()

emptyvalue(::Max{C}) where {C} = NamedTuple{(Symbol(C, :_max),)}((missing,))
fresh(::Max{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_max),intypes[C],typeof(max)}()

"""
    First(column::Symbol) -> Summarizer

The value of `column` in the first row folded, in `:{column}_first`, with
`column`'s element type (`missing` for no rows). In a sliding window it keeps
every value in the window, O(window) memory per key.
"""
struct First{C} <: GroupSummarizer end
First(column::Symbol) = First{column}()

emptyvalue(::First{C}) where {C} = NamedTuple{(Symbol(C, :_first),)}((missing,))
fresh(::First{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_first),intypes[C],typeof(keepfirst)}()

"""
    Last(column::Symbol) -> Summarizer

The value of `column` in the last row folded, in `:{column}_last`, with
`column`'s element type (`missing` for no rows).
"""
struct Last{C} <: GroupSummarizer end
Last(column::Symbol) = Last{column}()

emptyvalue(::Last{C}) where {C} = NamedTuple{(Symbol(C, :_last),)}((missing,))
fresh(::Last{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_last),intypes[C],typeof(keeplast)}()

# Min, Max and First slide the deque above: the TrackState they fold everywhere
# else has no inverse, but a window removes its rows oldest first, which the
# deque can.
freshwindowed(s::Union{Min,Max,First}, intypes::NamedTuple) =
    windowtrack(fresh(s, intypes))
windowtrack(::TrackState{C,N,T,F}) where {C,N,T,F} = WindowTrackState{C,N,T,F}()

# Last's windowed state: the newest value and a count of the rows in the window.
# Removing the oldest row can only change the last value by emptying the
# window, so a count is all eviction needs — a count rather than the last row's
# time, which could not tell tied rows apart. As TrackState's `seen` does, the
# count guards the value field, which is left undefined until the first row
# and stale once the window empties; `value` is only read with rows folded.
# Measured against the deque (which, under `keeplast`, would hold one value but
# still pop and push two vectors per row): 1.1-2.2 ns per row against 5.7-8.4,
# and 23 ms against 38 ms for a keyless summarizewindows of `Last` over the
# benchmark's million rows.
mutable struct WindowLastState{C,N,T} <: SummarizerState
    n::Int
    val::T
    WindowLastState{C,N,T}() where {C,N,T} = new{C,N,T}(0)
end

freshwindowed(::Last{C}, intypes::NamedTuple) where {C} =
    WindowLastState{C,Symbol(C, :_last),intypes[C]}()
fresh(::WindowLastState{C,N,T}) where {C,N,T} = WindowLastState{C,N,T}()
@inline fresh!(st::WindowLastState) = (st.n = 0; st)
@inline function update!(st::WindowLastState{C}, row) where {C}
    st.val = getproperty(row, C)
    st.n += 1
    return nothing
end
@inline downdate!(st::WindowLastState, row) = (st.n -= 1; nothing)
value(st::WindowLastState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))

"""
    FittedModel{P,M}

A fitted MLJ model, as [`FitModel`](@ref) emits it. It holds the `model`
(hyperparameters, of type `M`), the `fitresult` needed for prediction, and the
fit `report` (see [`modelreports`](@ref)). `P` is the tuple of predictor names,
which [`applymodels`](@ref) predicts from. Serialization, as in
[`writejls`](@ref), goes through MLJ's `save`/`restore`.
"""
struct FittedModel{P,M}
    model::M
    fitresult::Any
    report::Any
end

# Compact, so a frame of models prints as a table rather than as a dump of
# every fitresult.
Base.show(io::IO, fm::FittedModel{P}) where {P} =
    print(io, "FittedModel(", nameof(typeof(fm.model)), ", ", P, ")")

"""
    FitModel(model, predictors, response::Symbol; name = :model,
             verbosity = 0) -> Summarizer

Fits an MLJ model to the rows it summarizes, emitting a [`FittedModel`](@ref)
(`missing` for no rows). Requires MLJModelInterface: `using MLJ`, or an MLJ
model package (most models also need MLJBase, which `using MLJ` loads).

# Arguments
- `model`: an `MLJModelInterface.Model`; anything else is an `ArgumentError`.
- `predictors`: a column name, or a non-empty collection of distinct names.
  They reach the model as a column table of their own element types, without
  scientific-type coercion (coerce upstream with [`addcolumns`](@ref)), and
  `missing` is passed through.
- `response`: the response column, which may not be a predictor.

# Keywords
- `name = :model`: the output column.
- `verbosity = 0`: passed to the model's `fit`.

Rows are buffered and the model is fit each time a summary is emitted: once
under [`summarize`](@ref), per interval under [`intervalize`](@ref), per tick
and key under [`summarizewindows`](@ref), and after every row (quadratic)
under [`addsummarycolumns`](@ref). Windows re-fold for it, as it neither
combines nor inverts. Apply the models with [`applymodels`](@ref), or fit and
apply in one step with [`addpredictions`](@ref).
"""
struct FitModel{N,P,Y,M} <: Summarizer
    model::M
    verbosity::Int
end

function FitModel(model, predictors, response::Symbol; name::Symbol = :model,
    verbosity::Integer = 0)
    ismodel(model) || throw(
        ArgumentError(
            mljloaded() ?
            "FitModel model must be an MLJ model, got a $(typeof(model))" : MLJHINT,
        ),
    )
    ps = Tuple(Symbol(p) for p in predictors)
    isempty(ps) && throw(ArgumentError("FitModel requires at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "FitModel predictors must be unique, got $ps"))
    response in ps && throw(ArgumentError(
        "FitModel response $(repr(response)) is also a predictor"))
    return FitModel{name,ps,response,typeof(model)}(model, Int(verbosity))
end
# A lone name. A string is one name too, never iterated as a collection of
# one-character names.
FitModel(model, predictor::Union{Symbol,AbstractString}, response::Symbol;
    kwargs...) = FitModel(model, (Symbol(predictor),), response; kwargs...)

# The folded rows: one concretely typed vector per predictor plus one for the
# response, typed from the input schema like every other state. `fresh!`
# empties them in place, keeping their capacity — the re-fold windows zero a
# state per window — which is safe only because `value` fits on copies.
mutable struct FitModelState{N,P,Y,M,C<:NamedTuple,V<:AbstractVector} <:
               SummarizerState
    const model::M
    const verbosity::Int
    cols::C
    y::V
end

emptyvalue(::FitModel{N}) where {N} = NamedTuple{(N,)}((missing,))
function fresh(s::FitModel{N,P,Y,M}, intypes::NamedTuple) where {N,P,Y,M}
    cols = NamedTuple{P}(map(p -> Vector{intypes[p]}(), P))
    y = Vector{intypes[Y]}()
    return FitModelState{N,P,Y,M,typeof(cols),typeof(y)}(s.model, s.verbosity,
        cols, y)
end
fresh(st::FitModelState{N,P,Y,M,C,V}) where {N,P,Y,M,C,V} =
    FitModelState{N,P,Y,M,C,V}(st.model, st.verbosity,
        map(v -> similar(v, 0), st.cols), similar(st.y, 0))
@inline function fresh!(st::FitModelState)
    map(empty!, values(st.cols))
    empty!(st.y)
    return st
end
# `P` is a type parameter, so the map over it unrolls into one statically
# typed push per predictor column.
@inline function update!(st::FitModelState{N,P,Y}, row) where {N,P,Y}
    map(push!, values(st.cols), map(p -> getproperty(row, p), P))
    push!(st.y, getproperty(row, Y))
    return nothing
end
# The fit is opaque (an extension hook, returning Any), but the value's type is
# built from the type parameters, so the output column is concrete regardless.
function value(st::FitModelState{N,P,Y,M}) where {N,P,Y,M}
    fitresult, report = fitmodel(st.model, st.verbosity, map(copy, st.cols),
        copy(st.y))
    return NamedTuple{(N,),Tuple{FittedModel{P,M}}}((
        FittedModel{P,M}(st.model,
            fitresult, report),
    ))
end
function widenstate(st::FitModelState{N,P,Y,M},
    intypes::NamedTuple) where {N,P,Y,M}
    cols = NamedTuple{P}(map(p -> convert(Vector{intypes[p]}, st.cols[p]), P))
    y = convert(Vector{intypes[Y]}, st.y)
    return FitModelState{N,P,Y,M,typeof(cols),typeof(y)}(st.model, st.verbosity,
        cols, y)
end
