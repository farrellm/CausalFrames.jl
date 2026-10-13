# The summarizer interface. A summarization is
# split into an immutable Summarizer (the configuration) and a SummarizerState
# (the running state). The state is built from the input columns' element
# types, so its value fields, and so its output columns, are concretely typed
# by the input schema rather than by the values. That is what lets the folding
# loops in summarize.jl run type-stable behind a function barrier.

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
    inputcolumns(s::Summarizer) -> Union{Nothing, Tuple{Vararg{Symbol}}}

The input columns, besides `:time`, that `s`'s states read from a row in
[`update!`](@ref) and [`downdate!`](@ref); a row term counts under its name.
[`addrollingcolumns`](@ref) and [`summarizewindows`](@ref) store the rows they
summarize, and keep only these columns, the key columns and `:time`, so
[`fresh`](@ref) is given only their types. Defaults to `nothing`: `s` may read
any column, and every column is kept. A summarizer whose state reads no row (a
dependent one, or [`Count`](@ref)) declares `()`. A state folding rows through
[`barwindow`](@ref) declares what its inner summarizers read with
[`barinputcolumns`](@ref). An inner row term is evaluated before the rows are
stored only when the state hoists it (see [`rowterms`](@ref)); otherwise
`barinputcolumns` is `nothing`, and the state may instead declare the columns
the term's function reads. Declaring a column it does not read only costs
speed; leaving out one it reads is an error when it is read.
"""
inputcolumns(::Summarizer) = nothing

"""
    ColumnSpec

An input column as a summarizer constructor takes it: a column name (a
`Symbol`), or a *row term* `name => f`. A row term is a virtual input column
`name`, whose value in each row is `f(row)`, `row` being a `NamedTuple` of
the input row's columns. The summarizer reads it as it
would a column, so its output is named after `name` (`Sum(:range => f)` gives
`:range_sum`). The summarizing transforms compute each row term once per row,
alongside the input, and never add it to their output. See
[Row terms](@ref).

`Union{Symbol, Pair{Symbol, <:Base.Callable}}`.

```jldoctest
df = DataFrame(time = 1:3, high = [5.0, 6.0, 8.0], low = [4.0, 4.5, 5.0])
spread = r -> r.high - r.low   # bound once, so both summarizers share it
p = readtable(df) |>
    addsummarycolumns([Sum(:spread => spread), Mean(:spread => spread)])
DataFrame(load(Context(0, 10), p))

# output

3×5 DataFrame
 Row │ time   high     low      spread_sum  spread_mean
     │ Int64  Float64  Float64  Float64     Float64
─────┼──────────────────────────────────────────────────
   1 │     1      5.0      4.0         1.0      1.0
   2 │     2      6.0      4.5         2.5      1.25
   3 │     3      8.0      5.0         5.5      1.83333
```
"""
const ColumnSpec = Union{Symbol,Pair{Symbol,<:Base.Callable}}

colname(c::Symbol) = c
colname(p::Pair) = first(p)

# A summarizer and the row terms it reads, as `withterms` builds it for a
# constructor given a `name => f`. `prototypes` unwraps it before folding, so
# no state, tier or kernel ever sees one.
struct Termed{S<:Summarizer,T<:NamedTuple} <: Summarizer
    summarizer::S
    terms::T
end

emptyvalue(t::Termed) = emptyvalue(t.summarizer)

unterm(s::Summarizer) = s
unterm(t::Termed) = t.summarizer

"""
    rowterms(s::Summarizer) -> NamedTuple

Row terms the summarizing transforms evaluate for `s`'s states besides those
its constructor took through [`withterms`](@ref), as `name = f` functions of
the input row; each becomes a column of the rows passed to
[`update!`](@ref). Defaults to `(;)`. A summarizer whose state folds rows
through [`barwindow`](@ref) uses it to hoist its window's row terms out of the
window, so that its [`inputcolumns`](@ref) can name every column the window
reads. A term follows the rules of [Row terms](@ref): the same name must
always be the same function, across every summarizer of a call, and may not
name an input column.

```jldoctest
struct Trailing{S<:Summarizer} <: Summarizer
    inner::S
    n::Int
end
struct TrailingState{W<:SummarizerState} <: SummarizerState
    window::W
end
CausalFrames.emptyvalue(::Trailing) = (; trailing = missing)
CausalFrames.rowterms(s::Trailing) = CausalFrames.barterms(s.inner; hoistedby = s)
CausalFrames.inputcolumns(s::Trailing) =
    CausalFrames.barinputcolumns(s.inner; hoistedby = s)
CausalFrames.fresh(s::Trailing, intypes::NamedTuple) =
    TrailingState(CausalFrames.barwindow(s.inner, s.n, intypes; hoistedby = s))
CausalFrames.fresh(st::TrailingState) = TrailingState(CausalFrames.fresh(st.window))
CausalFrames.update!(st::TrailingState, row) = CausalFrames.update!(st.window, row)
function CausalFrames.value(st::TrailingState)
    v = CausalFrames.value(st.window)
    return NamedTuple{(:trailing,),Tuple{fieldtype(typeof(v), 1)}}((v[1],))
end

df = DataFrame(time = 1:4, high = [5.0, 6.0, 8.0, 7.0], low = [4.0, 4.5, 5.0, 6.0])
p = readtable(df) |>
    addsummarycolumns(Trailing(Mean(:spread => r -> r.high - r.low), 2))
DataFrame(load(Context(0, 10), p))

# output

4×4 DataFrame
 Row │ time   high     low      trailing
     │ Int64  Float64  Float64  Float64?
─────┼─────────────────────────────────────
   1 │     1      5.0      4.0  missing
   2 │     2      6.0      4.5        1.25
   3 │     3      8.0      5.0        2.25
   4 │     4      7.0      6.0        2.0
```
"""
rowterms(::Summarizer) = (;)
rowterms(t::Termed) = t.terms

# Add a `name => f` row term to `pairs`. The same name must always mean the
# same function (`===`, so two copies of one lambda literal differ).
function addterm!(pairs::Vector{Pair{Symbol,Any}}, name::Symbol, f, op::String)
    i = findfirst(p -> first(p) === name, pairs)
    if i === nothing
        push!(pairs, name => f)
    elseif last(pairs[i]) !== f
        throw(
            ArgumentError(
                "$op row term $(repr(name)) is defined by more than one function; " *
                "bind the function once and reuse it",
            ),
        )
    end
    return pairs
end

termtuple(pairs::Vector{Pair{Symbol,Any}}) =
    NamedTuple{Tuple(map(first, pairs))}(Tuple(map(last, pairs)))

"""
    withterms(s::Summarizer, specs::ColumnSpec...) -> Summarizer

`s`, carrying the row terms among `specs`, which are the [`ColumnSpec`](@ref)s
`s` was built from. A constructor taking a column calls it on the summarizer
built from the specs' names, as [`Sum`](@ref)'s does:
`Sum(column::ColumnSpec) = withterms(Sum{colname(column)}(), column)`.

With no row term among `specs` it returns `s` itself. A row term may not be
named `:time`, and two row terms with the same name must be the same function;
either is an `ArgumentError` naming `s`'s type.

```jldoctest
double = r -> 2r.x
s = CausalFrames.withterms(Sum{:d}(), :d => double)
(CausalFrames.rowterms(s) === (; d = double),
    CausalFrames.withterms(Sum{:x}(), :x) === Sum{:x}())

# output

(true, true)
```
"""
function withterms(s::Summarizer, specs::ColumnSpec...)
    any(p -> p isa Pair, specs) || return s
    op = string(nameof(typeof(s)))
    pairs = Pair{Symbol,Any}[]
    for p in specs
        p isa Pair || continue
        first(p) === :time &&
            throw(ArgumentError("$op row term may not be named :time"))
        addterm!(pairs, first(p), last(p), op)
    end
    return Termed(s, termtuple(pairs))
end

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

# Fold one row into (or, for group states, out of) every state in a tuple. Over
# a concrete tuple the foreach unrolls and dispatches statically.
@inline updateall!(states::Tuple, row) = foreach(st -> update!(st, row), states)
@inline downdateall!(states::Tuple, row) =
    foreach(st -> downdate!(st, row), states)

# Zero a state tuple in place, returning the tuple to use (as with `fresh!`, a
# state that can't be zeroed in place returns a new object). Over a concrete
# tuple the map unrolls; for the built-in states it allocates nothing.
@inline freshall!(states::Tuple) = map(fresh!, states)

@inline maybemissing(::Type{A}, M::Bool) where {A} = M ? Union{Missing,A} : A
