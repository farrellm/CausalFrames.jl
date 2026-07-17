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

Planned refinements will add structured subtypes: a monoid subtype (mergeable
state, enabling map-reduce) and a group subtype (invertible updates, enabling
efficient rolling windows), plus a mechanism for a summarizer to declare
dependent summarizers (e.g. variance depending on sum and sum of squares) so
shared work is computed once.
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

The current summary. A summarizer may produce several values; the keys of the
returned `NamedTuple` are the output column names, and its value types are the
element types of the columns produced.

Only ever called on a state that has folded at least one row — the summary of
no rows is [`emptyvalue`](@ref).
"""
function value end

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

# The element type Base.sum produces over a column of eltype T: small signed
# and unsigned integers widen to Int/UInt, everything else keeps its type. The
# accumulator is built at this width up front, so the folding loop is a plain
# `+` that is both type-stable and immune to the overflow that accumulating in
# the input's own type would risk.
sumtype(::Type{T}) where {T} = Base.promote_op(Base.add_sum, T, T)

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
`Symbol(column, :_sum, n)`, e.g. `SumPower(:x, 2)` produces `:x_sum2`. The
sum of no rows is `0`. The output column's element type follows the same rule
as [`Sum`](@ref), applied to the type of `column ^ n`.

`SumPower(column, 1)` produces `:x_sum1`, a distinct column from `Sum(:x)`'s
`:x_sum`.
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

emptyvalue(s::SumPower{C}) where {C} = NamedTuple{(Symbol(C, :_sum, s.power),)}((0,))
function fresh(s::SumPower{C}, intypes::NamedTuple) where {C}
    A = powertype(intypes[C], s.power)
    return SumPowerState{C,Symbol(C, :_sum, s.power),A}(s.power, convert(A, 0))
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
