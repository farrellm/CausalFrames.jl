# Summarizers fold rows into running state; the summarization transforms
# (summarize, summarizecycles, addsummarycolumns) drive them over the stream
# of chunks, carrying state across chunk boundaries.
#
# A summarization is split in two: an immutable Summarizer holding only the
# configuration, and a SummarizerState holding the running state. The state is
# built from the input columns' element types, so its value fields — and hence
# the element types of the columns it produces — are concrete. That split is
# what makes an output column's type a consequence of the input schema rather
# than an accident of the values, and it is what lets the folding loops run
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

# Min/Max/First/Last have no identity element, so each state carries a `seen`
# flag: it keeps "no rows folded in" distinct from a column holding missing or
# nothing. The value field is typed exactly like the input column — no
# Union{Missing,T} in the folding loop — and is left *undefined* until the
# first row; `seen` guards every read of it.
#
# This relies on `value` never being called on a state that has folded no
# rows, which the transforms guarantee: every key group and every cycle folds
# a row before emitting, a keyless summarize with at least one chunk has at
# least one row, and the no-rows case is answered by `emptyvalue` without ever
# building a state.

"""
    Min(column) -> Summarizer

Tracks the minimum of `column`. Produces the output column
`Symbol(column, :_min)`, e.g. `Min(:x)` produces `:x_min`, with the same
element type as `column`. The minimum of no rows is `missing`.
"""
struct Min{C} <: Summarizer end
Min(column::Symbol) = Min{column}()

mutable struct MinState{C,N,T} <: SummarizerState
    seen::Bool
    lo::T
    MinState{C,N,T}() where {C,N,T} = new{C,N,T}(false)
    MinState{C,N,T}(seen::Bool, lo) where {C,N,T} = new{C,N,T}(seen, lo)
end

emptyvalue(::Min{C}) where {C} = NamedTuple{(Symbol(C, :_min),)}((missing,))
fresh(::Min{C}, intypes::NamedTuple) where {C} =
    MinState{C,Symbol(C, :_min),intypes[C]}()
fresh(::MinState{C,N,T}) where {C,N,T} = MinState{C,N,T}()
@inline function update!(st::MinState{C}, row) where {C}
    v = getproperty(row, C)
    st.lo = st.seen ? min(st.lo, v) : v
    st.seen = true
    return nothing
end
value(st::MinState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.lo,))
function widenstate(st::MinState{C,N,T}, intypes::NamedTuple) where {C,N,T}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? MinState{C,N,T2}(true, convert(T2, st.lo)) :
        MinState{C,N,T2}()
end

"""
    Max(column) -> Summarizer

Tracks the maximum of `column`. Produces the output column
`Symbol(column, :_max)`, e.g. `Max(:x)` produces `:x_max`, with the same
element type as `column`. The maximum of no rows is `missing`.
"""
struct Max{C} <: Summarizer end
Max(column::Symbol) = Max{column}()

mutable struct MaxState{C,N,T} <: SummarizerState
    seen::Bool
    hi::T
    MaxState{C,N,T}() where {C,N,T} = new{C,N,T}(false)
    MaxState{C,N,T}(seen::Bool, hi) where {C,N,T} = new{C,N,T}(seen, hi)
end

emptyvalue(::Max{C}) where {C} = NamedTuple{(Symbol(C, :_max),)}((missing,))
fresh(::Max{C}, intypes::NamedTuple) where {C} =
    MaxState{C,Symbol(C, :_max),intypes[C]}()
fresh(::MaxState{C,N,T}) where {C,N,T} = MaxState{C,N,T}()
@inline function update!(st::MaxState{C}, row) where {C}
    v = getproperty(row, C)
    st.hi = st.seen ? max(st.hi, v) : v
    st.seen = true
    return nothing
end
value(st::MaxState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.hi,))
function widenstate(st::MaxState{C,N,T}, intypes::NamedTuple) where {C,N,T}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? MaxState{C,N,T2}(true, convert(T2, st.hi)) :
        MaxState{C,N,T2}()
end

"""
    First(column) -> Summarizer

Keeps the value of `column` from the first row folded in. Produces the output
column `Symbol(column, :_first)`, e.g. `First(:x)` produces `:x_first`, with
the same element type as `column`. The first of no rows is `missing`.
"""
struct First{C} <: Summarizer end
First(column::Symbol) = First{column}()

mutable struct FirstState{C,N,T} <: SummarizerState
    seen::Bool
    val::T
    FirstState{C,N,T}() where {C,N,T} = new{C,N,T}(false)
    FirstState{C,N,T}(seen::Bool, val) where {C,N,T} = new{C,N,T}(seen, val)
end

emptyvalue(::First{C}) where {C} = NamedTuple{(Symbol(C, :_first),)}((missing,))
fresh(::First{C}, intypes::NamedTuple) where {C} =
    FirstState{C,Symbol(C, :_first),intypes[C]}()
fresh(::FirstState{C,N,T}) where {C,N,T} = FirstState{C,N,T}()
@inline function update!(st::FirstState{C}, row) where {C}
    if !st.seen
        st.val = getproperty(row, C)
        st.seen = true
    end
    return nothing
end
value(st::FirstState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))
function widenstate(st::FirstState{C,N,T}, intypes::NamedTuple) where {C,N,T}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? FirstState{C,N,T2}(true, convert(T2, st.val)) :
        FirstState{C,N,T2}()
end

"""
    Last(column) -> Summarizer

Keeps the value of `column` from the most recent row folded in. Produces the
output column `Symbol(column, :_last)`, e.g. `Last(:x)` produces `:x_last`,
with the same element type as `column`. The last of no rows is `missing`.
"""
struct Last{C} <: Summarizer end
Last(column::Symbol) = Last{column}()

mutable struct LastState{C,N,T} <: SummarizerState
    seen::Bool
    val::T
    LastState{C,N,T}() where {C,N,T} = new{C,N,T}(false)
    LastState{C,N,T}(seen::Bool, val) where {C,N,T} = new{C,N,T}(seen, val)
end

emptyvalue(::Last{C}) where {C} = NamedTuple{(Symbol(C, :_last),)}((missing,))
fresh(::Last{C}, intypes::NamedTuple) where {C} =
    LastState{C,Symbol(C, :_last),intypes[C]}()
fresh(::LastState{C,N,T}) where {C,N,T} = LastState{C,N,T}()
@inline function update!(st::LastState{C}, row) where {C}
    st.val = getproperty(row, C)
    st.seen = true
    return nothing
end
value(st::LastState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))
function widenstate(st::LastState{C,N,T}, intypes::NamedTuple) where {C,N,T}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? LastState{C,N,T2}(true, convert(T2, st.val)) :
        LastState{C,N,T2}()
end

# --- shared plumbing -------------------------------------------------------

tosummarizers(s::Summarizer) = Summarizer[s]
tosummarizers(ss) = collect(Summarizer, ss)

tokeycolumns(::Nothing) = Symbol[]
tokeycolumns(k::Symbol) = Symbol[k]
tokeycolumns(ks) = collect(Symbol, ks)

# Deduplicate prototypes by output-name tuple (identical configurations
# collapse to one shared instance) and validate the surviving output names
# against :time, the key columns, and each other. The transforms then hold the
# survivors in a tuple, so the states derived from them are a concrete tuple
# too and the folding loops specialize on it.
function prototypes(ss::Vector{Summarizer}, keycols::Vector{Symbol})
    isempty(ss) && throw(ArgumentError("at least one summarizer is required"))
    protos = Summarizer[]
    seen = Set{Tuple{Vararg{Symbol}}}()
    used = Set{Symbol}()
    for s in ss
        outnames = keys(emptyvalue(s))
        outnames in seen && continue
        for n in outnames
            n === :time && throw(ArgumentError(
                "summarizer output column may not be named time"))
            n in keycols && throw(ArgumentError(
                "summarizer output column $n collides with a key column"))
            n in used && throw(ArgumentError(
                "output column $n is produced by more than one summarizer"))
            push!(used, n)
        end
        push!(seen, outnames)
        push!(protos, s)
    end
    return Tuple(protos)
end

# Input column element types, mirroring the row access in update!.
chunktypes(c::DataFrame) =
    NamedTuple{Tuple(Symbol.(names(c)))}(Tuple(eltype(col) for col in eachcol(c)))

# A source may infer a column's element type per chunk, so the state types
# track the promotion of every input type seen so far rather than trusting the
# first chunk.
promotetypes(::Nothing, b::NamedTuple) = b
promotetypes(a::NamedTuple, b::NamedTuple) =
    NamedTuple{keys(b)}(Tuple(haskey(a, k) ? promote_type(a[k], b[k]) : b[k]
                              for k in keys(b)))

newstates(protos::Tuple, intypes::NamedTuple) = map(s -> fresh(s, intypes), protos)
widenstates(states::Tuple, intypes::NamedTuple) =
    map(st -> widenstate(st, intypes), states)

# `stateprotos` must already be widened: it is what fixes the rebuilt table's
# value type, which a comprehension over an empty `groups` could not.
function widengroups(groups::Dict{K}, stateprotos::S,
                     intypes::NamedTuple) where {K,S}
    widened = Dict{K,S}()
    for (k, gs) in groups
        widened[k] = widenstates(gs, intypes)
    end
    return widened
end

# The group table's key type comes from the first row and its value type from
# the state prototypes, so the folding kernels specialize on a concrete Dict
# rather than the Dict{Any,Vector{Summarizer}} this would otherwise be.
newgroups(stateprotos::S, nt::NamedTuple, keynames::Val) where {S} =
    Dict{typeof(keyvalues(first(Tables.rows(nt)), keynames)),S}()

@inline summaryvalues(states::Tuple) = merge(map(value, states)...)
emptyvalues(protos::Tuple) = merge(map(emptyvalue, protos)...)

@inline summaryrow(t, states::Tuple) = merge((; time = t), summaryvalues(states))
@inline summaryrow(t, k::NamedTuple, states::Tuple) =
    merge((; time = t), k, summaryvalues(states))

# The key names ride in a Val so the group key's NamedTuple type — and hence
# the group table's Dict type — is known to the compiler.
@inline keyvalues(row, ::Val{KN}) where {KN} =
    NamedTuple{KN}(map(c -> getproperty(row, c), KN))

sortedgroups(groups) = sort!(collect(groups); by = kv -> Tuple(first(kv)))

# The row types the kernels emit. The state prototypes cannot simply be run
# through `value` to find out: Min/Max/First/Last leave their value field
# undefined until a row is folded in.
rowtype(::Type{T}, ::Type{S}) where {T,S} = Base.promote_op(summaryrow, T, S)
rowtype(::Type{T}, ::Type{K}, ::Type{S}) where {T,K,S} =
    Base.promote_op(summaryrow, T, K, S)
valuetype(::Type{S}) where {S} = Base.promote_op(summaryvalues, S)

# --- folding kernels -------------------------------------------------------
#
# Everything below is called once per chunk with concretely typed arguments,
# so each specializes on the state tuple and the chunk's column table and the
# per-row work compiles down to direct field access. The type-unstable setup —
# reading the schema, building or widening the states, turning the chunk into
# a column table — stays on the other side of that boundary, in the transforms.

function foldall!(states::Tuple, nt::NamedTuple)
    for row in Tables.rows(nt)
        foreach(st -> update!(st, row), states)
    end
    return nothing
end

function foldgroups!(groups::Dict{K,S}, stateprotos::S, nt::NamedTuple,
                     ::Val{KN}) where {K,S,KN}
    for row in Tables.rows(nt)
        states = get!(() -> map(fresh, stateprotos), groups, keyvalues(row, Val(KN)))
        foreach(st -> update!(st, row), states)
    end
    return nothing
end

# A cycle closes when a row with a later time arrives (causal), so the open
# cycle's state is carried across chunk boundaries and only closed by flush.
function foldcycles!(states::S, stateprotos::S, nt::NamedTuple,
                     cycletime) where {S<:Tuple}
    rows = rowtype(eltype(nt.time), S)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            cycletime === nothing ||
                push!(rows, summaryrow(something(cycletime), states))
            cycletime = t
            states = map(fresh, stateprotos)
        end
        foreach(st -> update!(st, row), states)
    end
    return rows, states, cycletime
end

function foldcyclesgrouped!(groups::Dict{K,S}, stateprotos::S, nt::NamedTuple,
                            cycletime, ::Val{KN}) where {K,S,KN}
    rows = rowtype(eltype(nt.time), K, S)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            cycletime === nothing || closecycle!(rows, groups, something(cycletime))
            cycletime = t
        end
        states = get!(() -> map(fresh, stateprotos), groups, keyvalues(row, Val(KN)))
        foreach(st -> update!(st, row), states)
    end
    return rows, cycletime
end

function closecycle!(rows, groups, t)
    for (k, states) in sortedgroups(groups)
        push!(rows, summaryrow(t, k, states))
    end
    empty!(groups)
    return rows
end

function foldrunning!(states::S, nt::NamedTuple, n::Int) where {S<:Tuple}
    vals = Vector{valuetype(S)}(undef, n)
    i = 0
    for row in Tables.rows(nt)
        foreach(st -> update!(st, row), states)
        vals[i += 1] = summaryvalues(states)
    end
    return vals
end

function foldrunninggrouped!(groups::Dict{K,S}, stateprotos::S, nt::NamedTuple,
                             n::Int, ::Val{KN}) where {K,S,KN}
    vals = Vector{valuetype(S)}(undef, n)
    i = 0
    for row in Tables.rows(nt)
        states = get!(() -> map(fresh, stateprotos), groups, keyvalues(row, Val(KN)))
        foreach(st -> update!(st, row), states)
        vals[i += 1] = summaryvalues(states)
    end
    return vals
end

# Per-chunk setup shared by the transforms: promote the schema, then build the
# states on the first chunk or widen them when the promotion has moved.
mutable struct Schema
    types::Union{Nothing,NamedTuple}
    widened::Bool
end
Schema() = Schema(nothing, false)

function observe!(sc::Schema, c::DataFrame)
    types = promotetypes(sc.types, chunktypes(c))
    sc.widened = sc.types !== nothing && types != sc.types
    sc.types = types
    return types
end

# --- summarization transforms ----------------------------------------------

"""
    summarize(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)

A transform summarizing the whole context: every input row is folded into the
summarizers and a single batch of rows is emitted at the context's end time
`stop`, dropping the input columns. `summarizers` is a [`Summarizer`](@ref)
or a collection of them; the output columns are `time`, the key columns, then
each summarizer's value columns.

Without `key` the output is exactly one row (the identity summary — e.g.
`count = 0` — when the input is empty). With `key` (a column name or
collection of column names) one row is emitted per unique key value, sorted
by key; an empty input yields no rows.
"""
function summarize(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            keyed = !isempty(keycols)
            schema = Schema()
            states = nothing        # keyless: the running state tuple
            stateprotos = nothing   # keyed: template for each group's states
            groups = nothing
            step = function (c)
                intypes = observe!(schema, c)
                nt = Tables.columntable(c)
                if keyed
                    if groups === nothing
                        stateprotos = newstates(protos, intypes)
                        groups = newgroups(stateprotos, nt, keynames)
                    elseif schema.widened
                        stateprotos = widenstates(stateprotos, intypes)
                        groups = widengroups(groups, stateprotos, intypes)
                    end
                    foldgroups!(groups, stateprotos, nt, keynames)
                else
                    states = states === nothing ? newstates(protos, intypes) :
                        schema.widened ? widenstates(states, intypes) : states
                    foldall!(states, nt)
                end
                return nothing
            end
            flush = function ()
                if !keyed
                    vals = states === nothing ? emptyvalues(protos) :
                        summaryvalues(states)
                    return DataFrame([merge((; time = ctx.stop), vals)])
                end
                groups === nothing && return nothing
                rows = [summaryrow(ctx.stop, k, gs) for (k, gs) in sortedgroups(groups)]
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            return chunkmap(step, p.run(ctx); flush = flush)
        end
    end
end

"""
    summarizecycles(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)

A transform summarizing each *cycle* — a maximal run of rows sharing one
timestamp — independently, with fresh state per cycle. For every cycle one
row is emitted at the cycle's time (per unique key value, sorted by key, when
`key` is given), dropping the input columns. A cycle spanning a chunk
boundary is summarized as a single cycle.
"""
function summarizecycles(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            keyed = !isempty(keycols)
            schema = Schema()
            states = nothing
            stateprotos = nothing
            groups = nothing
            cycletime = nothing
            step = function (c)
                intypes = observe!(schema, c)
                nt = Tables.columntable(c)
                # Both paths need the prototypes: a cycle takes fresh state.
                # The keyless path additionally carries the open cycle's state,
                # which the widening must carry over with it.
                if stateprotos === nothing
                    stateprotos = newstates(protos, intypes)
                    keyed ? (groups = newgroups(stateprotos, nt, keynames)) :
                        (states = newstates(protos, intypes))
                elseif schema.widened
                    stateprotos = widenstates(stateprotos, intypes)
                    keyed ? (groups = widengroups(groups, stateprotos, intypes)) :
                        (states = widenstates(states, intypes))
                end
                rows = if keyed
                    rs, cycletime = foldcyclesgrouped!(groups, stateprotos, nt,
                                                       cycletime, keynames)
                    rs
                else
                    rs, states, cycletime = foldcycles!(states, stateprotos, nt,
                                                        cycletime)
                    rs
                end
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            flush = function ()
                cycletime === nothing && return nothing
                rows = keyed ? closecycle!(
                        rowtype(typeof(cycletime), keytype(groups),
                                valtype(groups))[], groups, cycletime) :
                    [summaryrow(cycletime, states)]
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            return chunkmap(step, p.run(ctx); flush = flush)
        end
    end
end

"""
    addsummarycolumns(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)

A transform keeping all existing columns and appending each summarizer's
value columns, holding the running summary *after* that row has been folded
in. With `key` (a column name or collection of column names) a separate
running summary is kept per unique key value. State runs over the whole
window, carried across chunk boundaries. The output columns may not collide
with existing columns.
"""
function addsummarycolumns(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            keyed = !isempty(keycols)
            schema = Schema()
            states = nothing
            stateprotos = nothing
            groups = nothing
            checked = false   # collision check needs the schema: first chunk
            step = function (c)
                if !checked
                    for s in protos, n in keys(emptyvalue(s))
                        String(n) in names(c) && throw(ArgumentError(
                            "summary column $n collides with an existing column"))
                    end
                    checked = true
                end
                intypes = observe!(schema, c)
                nt = Tables.columntable(c)
                vals = if keyed
                    if groups === nothing
                        stateprotos = newstates(protos, intypes)
                        groups = newgroups(stateprotos, nt, keynames)
                    elseif schema.widened
                        stateprotos = widenstates(stateprotos, intypes)
                        groups = widengroups(groups, stateprotos, intypes)
                    end
                    foldrunninggrouped!(groups, stateprotos, nt, nrow(c), keynames)
                else
                    states = states === nothing ? newstates(protos, intypes) :
                        schema.widened ? widenstates(states, intypes) : states
                    foldrunning!(states, nt, nrow(c))
                end
                return hcat(c, DataFrame(vals))
            end
            return chunkmap(step, p.run(ctx))
        end
    end
end
