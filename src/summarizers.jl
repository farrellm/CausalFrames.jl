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

Structured subtypes refine what the state supports: a
[`MonoidSummarizer`](@ref)'s states combine associatively
([`combine!`](@ref)), and a [`GroupSummarizer`](@ref)'s updates are
additionally invertible ([`downdate!`](@ref)). `addrollingcolumns` selects
faster window algorithms when every summarizer it folds declares the
structure.
"""
abstract type Summarizer end

"""
    MonoidSummarizer <: Summarizer

A summarizer whose states form a monoid: [`combine!`](@ref) merges the states
of two adjacent, stream-ordered row ranges into the state of their
concatenation, associatively, with a [`fresh`](@ref) state as the identity.
`addrollingcolumns` exploits this to answer each window from a tree of
partial combinations — O(log window) per row — instead of re-folding every
window row.
"""
abstract type MonoidSummarizer <: Summarizer end

"""
    GroupSummarizer <: MonoidSummarizer

A monoid summarizer whose updates can also be undone: [`downdate!`](@ref)
removes a previously folded row. `addrollingcolumns` exploits this to slide
each window in O(1) amortized per row, subtracting the exiting rows from a
running state — unless [`isinvertible`](@ref) reports that the realized
accumulator type defeats the inverse (an absorbing value folded past
recovery), in which case it falls back to the monoid tree.
"""
abstract type GroupSummarizer <: MonoidSummarizer end

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
    fresh!(st::SummarizerState) -> SummarizerState

Zero `st` in place and return it — the in-place counterpart of the one-argument
[`fresh`](@ref). Callers must use the returned value rather than assume `st`
was mutated: the default implementation is `fresh(st)`, so a state that cannot
be zeroed in place (an immutable one) simply returns a new object, and every
existing summarizer keeps working without implementing this at all.

Implementing it is a pure optimization, and worth it for any state on a hot
path: the transforms zero a state tuple per cycle
([`summarizecycles`](@ref)), per window query (the `addrollingcolumns` tree
mode), and per window per row (its re-fold mode), so an allocating `fresh`
there costs one heap allocation per state per row.

The zeroed state must be indistinguishable from `fresh(st)` through the rest of
the interface. The one exception is the same one [`fresh`](@ref) already
carries: a state whose value field is only meaningful once a row has been
folded in (`Min`/`Max`/`First`/`Last`) may leave a stale value behind, because
[`value`](@ref) is never called on a state that has folded no rows.
"""
fresh!(st::SummarizerState) = fresh(st)

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
`intypes`, carrying the accumulated value over. A source may hand a column a
different element type from one chunk to the next, so a column can be `Int` in
one chunk and `Float64` in the next; the transforms promote the types they have
seen and call `widenstate` when that promotion changes something.

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

"""
    combine!(dest::SummarizerState, a::SummarizerState, b::SummarizerState)

Overwrite `dest` with the combination of `a` and `b`: the state that folding
`a`'s rows and then `b`'s rows into a fresh state would produce. Required of
a [`MonoidSummarizer`](@ref)'s states, with the monoid laws: combination is
associative, and a [`fresh`](@ref) state is the identity on either side.
Callers guarantee that every row folded into `a` precedes every row folded
into `b` in stream order — which is what lets order-sensitive summarizers
like `First` and `Last` combine. All three states are of the same concrete
type, and `dest` may alias `a` or `b`, so an implementation reads its inputs
before writing.
"""
function combine! end

"""
    downdate!(st::SummarizerState, row)

Remove one previously folded row from the state — the inverse of
[`update!`](@ref). Required of a [`GroupSummarizer`](@ref)'s states. The
inverse is exact when the accumulator arithmetic is (integer sums); the
floating-point sum accumulators use compensated summation with NaN and ±Inf
terms counted separately, so nonfinite rows subtract away exactly and finite
ones leave only the small compensated round-off. `missing` terms are likewise
counted rather than folded in (see the Optional* accumulator states), so a
missing row also subtracts away exactly — the count balances — leaving the
accumulator invertible.
"""
function downdate! end

"""
    isinvertible(st::SummarizerState) -> Bool

Whether [`downdate!`](@ref) actually inverts [`update!`](@ref) for this
state's realized accumulator type. Defaults to `true`. The sum-family
accumulators keep NaN, ±Inf, and `missing` terms out of the running total and
count them instead, so they stay invertible; a state that folds an absorbing
value into an unrecoverable running total returns `false`. `addrollingcolumns`
consults this when choosing the running-state window algorithm.
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
# the same way (the Optional* states, over the non-missing type), so they too
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

# The sum family (Sum, SumPower, DotProduct) shares one plain and one
# compensated accumulator state over a *term functor* — the TrackState
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
# pair) either operand missing. Only instantiated for the Optional* states,
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

# N names the output column and A is the realized accumulator type, as on
# every other state; T is the term functor's type, so update! inlines the
# term computation statically.
mutable struct AccumState{N,A,T} <: SummarizerState
    term::T
    total::A
end

mutable struct CompensatedAccumState{N,A<:AbstractFloat,T} <: SummarizerState
    term::T
    acc::Compensated{A}
end

# A `missing` input term absorbs a running sum and cannot be subtracted back
# out, which would force a rolling window off the O(1) running path onto the
# tree. So, exactly as the compensated state counts nonfinite floats, these two
# states count the missing terms instead of folding them in: the accumulation
# lives at the *non-missing* type A (`total`/`acc`, flat — no Union in the hot
# field), only finite/present terms enter it, `missings` tracks the rest, and
# `value` reports `missing` whenever that count is positive. The count balances
# under `downdate!`, so the accumulator stays invertible and a missing row
# recovers once it leaves the window. Used only when a source column admits
# Missing; `OptionalCompensatedAccumState` also keeps the compensated state's
# nonfinite counters.
mutable struct OptionalAccumState{N,A,T} <: SummarizerState
    term::T
    total::A
    missings::Int
end

mutable struct OptionalCompensatedAccumState{N,A<:AbstractFloat,T} <: SummarizerState
    term::T
    acc::Compensated{A}
    missings::Int
end

# Shared constructor behind the sum family's fresh methods: a Missing-admitting
# accumulator type folds into the counting Optional* states over the
# non-missing type; otherwise compensable types get the Neumaier state. (The
# `Union{}` guard keeps a pathological all-Missing column on the old path.)
function accumfresh(term, N::Symbol, ::Type{A}) where {A}
    Missing <: A && nonmissingtype(A) !== Union{} &&
        return optionalfresh(term, N, nonmissingtype(A))
    compensable(A) &&
        return CompensatedAccumState{N,A,typeof(term)}(term, compzero(A))
    return AccumState{N,A,typeof(term)}(term, convert(A, 0))
end

# Fresh Optional* state over the non-missing accumulator type A.
function optionalfresh(term, N::Symbol, ::Type{A}) where {A}
    compensable(A) && return OptionalCompensatedAccumState{N,A,typeof(term)}(
        term, compzero(A), 0)
    return OptionalAccumState{N,A,typeof(term)}(term, convert(A, 0), 0)
end

fresh(st::AccumState{N,A,T}) where {N,A,T} =
    AccumState{N,A,T}(st.term, convert(A, 0))
@inline fresh!(st::AccumState{N,A}) where {N,A} = (st.total = convert(A, 0); st)
@inline update!(st::AccumState{N,A}, row) where {N,A} =
    (st.total += termvalue(st.term, A, row); nothing)
@inline downdate!(st::AccumState{N,A}, row) where {N,A} =
    (st.total -= termvalue(st.term, A, row); nothing)
combine!(dest::AccumState{N,A,T}, a::AccumState{N,A,T},
    b::AccumState{N,A,T}) where {N,A,T} =
    (dest.total = a.total + b.total; nothing)
value(st::AccumState{N,A}) where {N,A} = NamedTuple{(N,),Tuple{A}}((st.total,))
# A later chunk can widen a non-missing accumulator into a Missing-admitting
# type; that promotes it to the counting Optional* state (no missing folded in
# yet, so missings = 0) rather than a poisoned plain state.
function widenstate(st::AccumState{N,A,T}, intypes::NamedTuple) where {N,A,T}
    A2 = acctype(st.term, intypes)
    A2 === A && return st
    Missing <: A2 && nonmissingtype(A2) !== Union{} &&
        return optionalfrom(st.term, Val(N), nonmissingtype(A2), st.total, 0)
    compensable(A2) && return CompensatedAccumState{N,A2,T}(
        st.term, compadd(compzero(A2), convert(A2, st.total)))
    return AccumState{N,A2,T}(st.term, convert(A2, st.total))
end

fresh(st::CompensatedAccumState{N,A,T}) where {N,A,T} =
    CompensatedAccumState{N,A,T}(st.term, compzero(A))
@inline fresh!(st::CompensatedAccumState{N,A}) where {N,A} =
    (st.acc = compzero(A); st)
@inline update!(st::CompensatedAccumState{N,A}, row) where {N,A} =
    (st.acc = compadd(st.acc, termvalue(st.term, A, row)); nothing)
@inline downdate!(st::CompensatedAccumState{N,A}, row) where {N,A} =
    (st.acc = compsub(st.acc, termvalue(st.term, A, row)); nothing)
combine!(dest::CompensatedAccumState{N,A,T}, a::CompensatedAccumState{N,A,T},
    b::CompensatedAccumState{N,A,T}) where {N,A,T} =
    (dest.acc = compmerge(a.acc, b.acc); nothing)
value(st::CompensatedAccumState{N,A}) where {N,A} =
    NamedTuple{(N,),Tuple{A}}((compvalue(st.acc),))
function widenstate(st::CompensatedAccumState{N,A,T},
    intypes::NamedTuple) where {N,A,T}
    A2 = acctype(st.term, intypes)
    A2 === A && return st
    a = st.acc
    if Missing <: A2 && nonmissingtype(A2) !== Union{}
        Ann = nonmissingtype(A2)
        return compensable(Ann) ?
               OptionalCompensatedAccumState{N,Ann,T}(st.term,
            widencomp(Ann, a), 0) :
               OptionalAccumState{N,Ann,T}(st.term, convert(Ann, compvalue(a)), 0)
    end
    compensable(A2) && return CompensatedAccumState{N,A2,T}(
        st.term, widencomp(A2, a))
    return AccumState{N,A2,T}(st.term, convert(A2, compvalue(a)))
end

# Reinterpret a Compensated at a wider float type, carrying its running pair
# and its nonfinite counters unchanged.
widencomp(::Type{A2}, a::Compensated) where {A2} =
    Compensated{A2}(convert(A2, a.total), convert(A2, a.comp),
        a.nans, a.posinf, a.neginf)

# Build an Optional* state from a scalar total (used when promoting a plain,
# uncompensated accumulator that carries no compensation/nonfinite state).
function optionalfrom(term, ::Val{N}, ::Type{A2}, total, missings::Int) where {N,A2}
    T = typeof(term)
    compensable(A2) && return OptionalCompensatedAccumState{N,A2,T}(
        term, compadd(compzero(A2), convert(A2, total)), missings)
    return OptionalAccumState{N,A2,T}(term, convert(A2, total), missings)
end

# The Optional* interface: fold only present terms into the accumulation and
# count the missing ones; `value` is `missing` while any missing term is live.
# The value's field type is the static Union{Missing,A} — a runtime `typeof`
# would let a missing window collapse a dependent summarizer's output type.
fresh(st::OptionalAccumState{N,A,T}) where {N,A,T} =
    OptionalAccumState{N,A,T}(st.term, convert(A, 0), 0)
@inline fresh!(st::OptionalAccumState{N,A}) where {N,A} =
    (st.total = convert(A, 0); st.missings = 0; st)
@inline update!(st::OptionalAccumState{N,A}, row) where {N,A} =
    (
        termmissing(st.term, row) ? (st.missings += 1) :
        (st.total += termvalue(st.term, A, row)); nothing)
@inline downdate!(st::OptionalAccumState{N,A}, row) where {N,A} =
    (
        termmissing(st.term, row) ? (st.missings -= 1) :
        (st.total -= termvalue(st.term, A, row)); nothing)
function combine!(dest::OptionalAccumState{N,A,T}, a::OptionalAccumState{N,A,T},
    b::OptionalAccumState{N,A,T}) where {N,A,T}
    dest.total = a.total + b.total
    dest.missings = a.missings + b.missings
    return nothing
end
value(st::OptionalAccumState{N,A}) where {N,A} =
    NamedTuple{(N,),Tuple{Union{Missing,A}}}((st.missings > 0 ? missing : st.total,))
function widenstate(st::OptionalAccumState{N,A,T}, intypes::NamedTuple) where {N,A,T}
    A2 = nonmissingtype(acctype(st.term, intypes))
    A2 === A && return st
    return optionalfrom(st.term, Val(N), A2, st.total, st.missings)
end

fresh(st::OptionalCompensatedAccumState{N,A,T}) where {N,A,T} =
    OptionalCompensatedAccumState{N,A,T}(st.term, compzero(A), 0)
@inline fresh!(st::OptionalCompensatedAccumState{N,A}) where {N,A} =
    (st.acc = compzero(A); st.missings = 0; st)
@inline update!(st::OptionalCompensatedAccumState{N,A}, row) where {N,A} =
    (
        termmissing(st.term, row) ? (st.missings += 1) :
        (st.acc = compadd(st.acc, termvalue(st.term, A, row))); nothing)
@inline downdate!(st::OptionalCompensatedAccumState{N,A}, row) where {N,A} =
    (
        termmissing(st.term, row) ? (st.missings -= 1) :
        (st.acc = compsub(st.acc, termvalue(st.term, A, row))); nothing)
function combine!(dest::OptionalCompensatedAccumState{N,A,T},
    a::OptionalCompensatedAccumState{N,A,T},
    b::OptionalCompensatedAccumState{N,A,T}) where {N,A,T}
    dest.acc = compmerge(a.acc, b.acc)
    dest.missings = a.missings + b.missings
    return nothing
end
value(st::OptionalCompensatedAccumState{N,A}) where {N,A} =
    NamedTuple{(N,),Tuple{Union{Missing,A}}}((
        st.missings > 0 ? missing :
        compvalue(st.acc),
    ))
function widenstate(st::OptionalCompensatedAccumState{N,A,T},
    intypes::NamedTuple) where {N,A,T}
    A2 = nonmissingtype(acctype(st.term, intypes))
    A2 === A && return st
    return compensable(A2) ?
           OptionalCompensatedAccumState{N,A2,T}(st.term, widencomp(A2, st.acc),
        st.missings) :
           OptionalAccumState{N,A2,T}(st.term, convert(A2, compvalue(st.acc)),
        st.missings)
end

"""
    Count() -> Summarizer

Counts rows. Produces the output column `:count`, of type `Int`.
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
    CountDistinct(column) -> Summarizer

Counts the distinct values of `column`. Produces the output column
`Symbol(column, :_countdistinct)`, e.g. `CountDistinct(:x)` produces
`:x_countdistinct`, of type `Int`. The distinct count of no rows is `0`.

`missing` counts as a value like any other, so a column holding `1`, `missing`,
`1` has two distinct values, and the output column is `Int` rather than
`Union{Missing, Int}`. This is the one place the accumulating summarizers'
poisoning rule does not apply, and deliberately: a sum with a `missing` term is
unknowable, while a distinct count never is — you know exactly how many
distinct things you saw. SQL's `count(DISTINCT x)` skips nulls instead;
`filterrows(r -> !ismissing(r.x))` upstream recovers that reading.

Unlike every other summarizer, whose state is O(1), this one holds the distinct
values it has seen: folding `n` rows costs O(distinct) memory, and a rolling
window pays that per window per key.
"""
struct CountDistinct{C} <: MonoidSummarizer end
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

"""
    Sum(column) -> Summarizer

Sums `column`. Produces the output column `Symbol(column, :_sum)`, e.g.
`Sum(:x)` produces `:x_sum`. The sum of no rows is `0`.

The output column's element type is the one `Base.sum` would produce: small
signed and unsigned integers widen (`Int32` sums to `Int64`), everything else
keeps its type (`Float32` sums to `Float32`).

Floating-point accumulators use compensated (Neumaier) summation and count
NaN and ±Inf inputs separately: results match `Base.sum`, and a rolling
window recovers exactly once a nonfinite row leaves the window.
"""
struct Sum{C} <: GroupSummarizer end
Sum(column::Symbol) = Sum{column}()

emptyvalue(::Sum{C}) where {C} = NamedTuple{(Symbol(C, :_sum),)}((0,))
fresh(::Sum{C}, intypes::NamedTuple) where {C} =
    accumfresh(ColumnTerm{C}(), Symbol(C, :_sum), sumtype(intypes[C]))

"""
    SumPower(column, n) -> Summarizer

Sums `column` raised to the power `n`. Produces the output column
`Symbol(column, :_sumpower_, n)`, e.g. `SumPower(:x, 2)` produces
`:x_sumpower_2`. The sum of no rows is `0`. The output column's element type
follows the same rule as [`Sum`](@ref), applied to the type of `column ^ n`.
Each term is formed in the accumulator's (widened) type before raising to
the power, so a per-row power cannot overflow the way raising in the input
column's own type would. Floating-point accumulators use compensated
summation with NaN and ±Inf *terms* (the value after raising to the power)
counted separately, as in [`Sum`](@ref).

`SumPower(column, 1)` produces `:x_sumpower_1`, a distinct column from
`Sum(:x)`'s `:x_sum`.
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
# rounded square, while the runtime `^` is 1 ULP off it for a small fraction of
# inputs whose square lands near underflow (66 of 500k random Float64 bit
# patterns). The specialization is the more accurate of the two there — but it
# is a change, so it is stated rather than glossed. What the compensated states
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
    Product(column) -> Summarizer

Multiplies `column`. Produces the output column `Symbol(column, :_product)`,
e.g. `Product(:x)` produces `:x_product`. The product of no rows is `1`.

The output column's element type is the one `Base.prod` would produce: small
signed and unsigned integers widen (`Int32` multiplies to `Int64`), everything
else keeps its type (`Float32` stays `Float32`). Like [`Sum`](@ref), the
accumulator is built at that width up front.
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
    DotProduct(a, b) -> Summarizer

Sums the elementwise product of columns `a` and `b`. Produces the output
column `Symbol(a, :_, b, :_dotproduct)`, e.g. `DotProduct(:x, :y)` produces
`:x_y_dotproduct`. The dot product of no rows is `0`.

The output column's element type is `Base.sum` applied to the type of `a * b`,
so it widens the same way [`Sum`](@ref) does. Each term is formed in the
accumulator's (widened) type, so a per-row product cannot overflow the way
multiplying in the input columns' own types would. Floating-point
accumulators use compensated summation with NaN and ±Inf *terms* (the
per-row product, so `Inf * 0.0` counts as a NaN term) counted separately, as
in [`Sum`](@ref).

The value is symmetric, so the fold happens under one canonical (sorted)
argument order: `DotProduct(:y, :x)` still produces `:y_x_dotproduct`, but it
folds nothing of its own — it is a dependent summarizer over
`DotProduct(:x, :y)`, renaming that value. Asking both ways in one call, or
asking one way beside a [`Covariance`](@ref) or [`LinearRegression`](@ref) that
needs the same product, therefore costs one accumulator rather than two.
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
    Mean(column) -> Summarizer

The mean of `column`. Produces the output column `Symbol(column, :_mean)`,
e.g. `Mean(:x)` produces `:x_mean`. The mean of no rows is `missing`.

A dependent summarizer, computed as [`Sum`](@ref)`(column)` divided by
[`Count`](@ref) — those are folded alongside it but appear in the output only
if requested themselves. The output column's element type is the division's
result (`Int` input divides to `Float64`, `Float32` stays `Float32`).
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
    LinearRegression(predictors, response; intercept = true, name = nothing)

Ordinary least squares of `response` on `predictors` — a collection of column
names, or a single column name. With `intercept` (the default) the fit carries
a constant term.

# Output columns

`name`, when given, prefixes every output column as `Symbol(name, :_, base)`;
the base names are:

| # | base column | meaning |
|---|---|---|
| 1 | `n` | rows folded in |
| 2 | `r2` | coefficient of determination |
| 3 | `stderr` | residual standard error |
| 4 | `intercept_beta` | the constant term, omitted when `intercept = false` |
| 5 | `intercept_tstat` | its t statistic, omitted when `intercept = false` |
| 6.. | `Symbol(p, :_beta)`, `Symbol(p, :_tstat)` | per predictor `p`, in the order given |

so `2K + 5` columns for `K` predictors with an intercept, `2K + 3` without:

```julia
LinearRegression([:x, :z], :y)
# :n, :r2, :stderr, :intercept_beta, :intercept_tstat,
# :x_beta, :x_tstat, :z_beta, :z_tstat

LinearRegression([:x, :z], :y; name = :m1)
# :m1_n, :m1_r2, :m1_stderr, :m1_intercept_beta, :m1_intercept_tstat,
# :m1_x_beta, :m1_x_tstat, :m1_z_beta, :m1_z_tstat
```

The un-prefixed defaults are deliberately grabby: `n`, `r2`, and `stderr` say
nothing about the model, so **two regressions in one call need distinct
`name`s** — without them they collide on those three columns even when their
predictors differ entirely, and the collision is an error. A regression and the
same regression with `intercept` flipped likewise cannot share a `name`. A
predictor named `intercept` collides with the constant term's own pair and is
rejected by the constructor; the other model-level names take no suffix, so a
predictor named `n`, `r2`, or `stderr` is fine.

The `2K + 4` statistic columns share one element type, the computation's result
(`Float64` for integer input, `Float32` for `Float32`), and a
`Missing`-admitting input makes all of them `Union{Missing,…}`. `n` is
separately `Int`, exactly as [`Count`](@ref)'s `:count` is, and is never
`missing`.

The regression of no rows is `missing` in every statistic and `0` in `n`; so is
any window holding a `missing` in a predictor or the response. A rank-deficient
system — collinear predictors, or `n ≤ K` (`n ≤ K + 1` with an intercept) —
gives `NaN` statistics rather than raising, as `Correlation` does for a single
row. With no residual degrees of freedom left (`n = K + 1`, or `n = K` without
an intercept) the coefficients and `r2` are the exact fit but `stderr` and every
t statistic are `NaN`, and a constant response makes `r2` `NaN`. `n` is the
honest row count throughout, so a poisoned fit still says how much data it saw.

# Sharing

A dependent summarizer over [`Count`](@ref), [`SumPower`](@ref)`(·, 2)`,
[`DotProduct`](@ref), and — only when `intercept` — [`Sum`](@ref): the whole
normal-equations system and every statistic drawn from it are functions of
those. They are folded alongside the regression but appear in the output only
if requested themselves, and because dependencies deduplicate by output name,
two regressions over overlapping columns fold each cross product exactly once.
Cross products are requested under the canonical (sorted) argument order every
symmetric summarizer shares, so a `Covariance` or `DotProduct` requested
separately shares the accumulator whichever way round it is written. A squared
term is requested as `SumPower(c, 2)` — the name `Variance`, `Std`, and
`Correlation` already depend on — so a regression run beside them shares that
too.

With an intercept the system is centered on the column means, the multivariate
form of the identity [`Covariance`](@ref) uses: better conditioned, and one
dimension smaller than carrying a column of ones. Without one, `r2` is the
uncentered coefficient of determination, as is conventional for a
no-intercept fit.

Unlike every other dependent summarizer, a multiple regression allocates: `K ≥
2` builds a `K × K` workspace per emitted row. Simple regression (`K = 1`, the
common case) runs a closed form over scalars and allocates nothing.
"""
struct LinearRegression{P,Y} <: GroupSummarizer
    intercept::Bool
    name::Union{Nothing,Symbol}
end

function LinearRegression(predictors, response::Symbol;
    intercept::Bool = true, name::Union{Nothing,Symbol} = nothing)
    ps = Tuple(Symbol(p) for p in predictors)
    isempty(ps) &&
        throw(ArgumentError("LinearRegression needs at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "LinearRegression predictors must be distinct, got $ps"))
    # A predictor named `intercept` produces the constant term's own pair of
    # columns. Caught here rather than left to the NamedTuple constructor,
    # whose "duplicate field name" says nothing about which summarizer built
    # it. The check is on the whole name set rather than that one case, so it
    # stays honest if the naming scheme grows.
    outs = regnames(ps, name, intercept)
    allunique(outs) || throw(ArgumentError(
        "LinearRegression output columns are not distinct: $outs"))
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

# The derived states are fieldless — the summary is computed from the
# dependencies' values at emission time — so combining and downdating them is
# a no-op; their group structure is exactly that of their (transitively all
# group) dependencies, which the transforms fold alongside them.
const DerivedState = Union{AliasState,MomentState,MeanState,VarianceState,
    StdState,CovarianceState,CorrelationState,LinearRegressionState}
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

"""
    Min(column) -> Summarizer

Tracks the minimum of `column`. Produces the output column
`Symbol(column, :_min)`, e.g. `Min(:x)` produces `:x_min`, with the same
element type as `column`. The minimum of no rows is `missing`.
"""
struct Min{C} <: MonoidSummarizer end
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
struct Max{C} <: MonoidSummarizer end
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
struct First{C} <: MonoidSummarizer end
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
struct Last{C} <: MonoidSummarizer end
Last(column::Symbol) = Last{column}()

emptyvalue(::Last{C}) where {C} = NamedTuple{(Symbol(C, :_last),)}((missing,))
fresh(::Last{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_last),intypes[C],typeof(keeplast)}()

"""
    FittedModel{P,M}

A fitted MLJ model, as [`FitModel`](@ref) emits it: the `model` (its
hyperparameters, of type `M`), the `fitresult` its `fit` returned — everything
prediction needs — and the fit's `report` of diagnostics (see
[`modelreports`](@ref)). `P` is the tuple of predictor column names it was fit
on, which is how [`applymodels`](@ref) finds the columns to predict from, so a
table of fitted models is self-describing. Serializing one, as
[`writejls`](@ref) does, routes the fitresult through MLJ's `save`/`restore`,
so models wrapping foreign resources survive the round trip.
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
    FitModel(model, predictors, response; name = :model,
             verbosity = 0) -> Summarizer

Fits an MLJ `model` — any `MLJModelInterface.Model`, so anything in MLJ's model
registry — regressing `response` on `predictors` (a column name or a collection
of them) over the rows it folds. Produces one output column, `name`, holding a
[`FittedModel`](@ref); the summary of no rows is `missing`. Needs
MLJModelInterface loaded — `using MLJ`, or any MLJ model package; most model
implementations also need MLJBase, which `using MLJ` loads.

The folded rows are buffered and the model is fit when a summary is emitted, so
a fit's cost follows the host transform: once per window under
[`summarize`](@ref), per interval under [`intervalize`](@ref), per tick (and
key) under [`summarizewindows`](@ref) — the natural hosts. Under
[`addsummarycolumns`](@ref) it refits after every row, over everything seen so
far, which is quadratic. `verbosity` is passed to the model's `fit`.

The predictors reach the model as a column table of the input's own element
types, with no scientific-type coercion — coerce upstream with
[`addcolumns`](@ref) where a model needs it — and `missing` values are passed
through as they are. The model is handed its own copy of the rows, so a fitted
model that keeps its training data is never disturbed by later fits.

A fit neither combines nor inverts, so `FitModel` is a plain `Summarizer`:
windowed transforms re-fold each window for it. [`addpredictions`](@ref) fits
and applies models over a rolling window in one step.

MLJ exports the scientific type `Count`, so with both `using MLJ` and
`using CausalFrames` the [`Count`](@ref) summarizer must be written
`CausalFrames.Count()`.
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
            "FitModel needs an MLJ model; got a $(typeof(model))" : MLJHINT,
        ),
    )
    ps = Tuple(Symbol(p) for p in predictors)
    isempty(ps) && throw(ArgumentError("FitModel needs at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "FitModel predictors must be distinct, got $ps"))
    response in ps && throw(ArgumentError(
        "FitModel response $response is also a predictor"))
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
