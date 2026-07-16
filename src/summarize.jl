# Summarizers fold rows into running state; the summarization transforms
# (summarize, summarizecycles, addsummarycolumns) drive them over the stream
# of chunks, carrying state across chunk boundaries.

"""
    Summarizer

Abstract supertype for summarizations. A concrete summarizer instance holds
the running state of one summarization (typically over a particular column or
columns) and implements:

- [`fresh`](@ref)`(s)` — a new instance with the same configuration and zero
  state;
- [`update!`](@ref)`(s, row)` — fold one row into the state;
- [`value`](@ref)`(s)` — the current summary as a `NamedTuple` whose keys are
  the output column names.

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
    fresh(s::Summarizer) -> Summarizer

A new summarizer with the same configuration as `s` but zero state. The
summarization transforms treat the summarizers they are given as prototypes
and work on fresh copies, so one prototype serves many key groups and the
caller's instance is never mutated.
"""
function fresh end

"""
    update!(s::Summarizer, row)

Fold one row into the summarizer's state. `row` is a map-like row object
supporting `row.name` and `row[:name]` access, including `row.time`, so a
summarizer may read whichever columns it needs.
"""
function update! end

"""
    value(s::Summarizer) -> NamedTuple

The current summary. A summarizer may produce several values; the keys of the
returned `NamedTuple` are the output column names.
"""
function value end

"""
    Count() -> Summarizer

Counts rows. Produces the output column `:count`.
"""
mutable struct Count <: Summarizer
    n::Int
end
Count() = Count(0)

fresh(::Count) = Count()
update!(s::Count, row) = (s.n += 1; nothing)
value(s::Count) = (; count = s.n)

"""
    Sum(column) -> Summarizer

Sums `column`. Produces the output column `Symbol(column, :_sum)`, e.g.
`Sum(:x)` produces `:x_sum`. The sum of no rows is `0`.
"""
mutable struct Sum <: Summarizer
    column::Symbol
    total::Any
end
Sum(column::Symbol) = Sum(column, 0)

fresh(s::Sum) = Sum(s.column)
update!(s::Sum, row) = (s.total += row[s.column]; nothing)
value(s::Sum) = NamedTuple{(Symbol(s.column, :_sum),)}((s.total,))

"""
    SumPower(column, n) -> Summarizer

Sums `column` raised to the power `n`. Produces the output column
`Symbol(column, :_sum, n)`, e.g. `SumPower(:x, 2)` produces `:x_sum2`. The
sum of no rows is `0`.

`SumPower(column, 1)` produces `:x_sum1`, a distinct column from `Sum(:x)`'s
`:x_sum`.
"""
mutable struct SumPower <: Summarizer
    column::Symbol
    power::Int
    total::Any
end
SumPower(column::Symbol, power::Integer) = SumPower(column, Int(power), 0)

fresh(s::SumPower) = SumPower(s.column, s.power)
update!(s::SumPower, row) = (s.total += row[s.column]^s.power; nothing)
value(s::SumPower) = NamedTuple{(Symbol(s.column, :_sum, s.power),)}((s.total,))

# Min/Max/First/Last have no identity element, so each carries a `seen` flag:
# it keeps "no rows folded in" (which yields missing) distinct from a column
# holding missing or nothing.

"""
    Min(column) -> Summarizer

Tracks the minimum of `column`. Produces the output column
`Symbol(column, :_min)`, e.g. `Min(:x)` produces `:x_min`. The minimum of no
rows is `missing`.
"""
mutable struct Min <: Summarizer
    column::Symbol
    seen::Bool
    lo::Any
end
Min(column::Symbol) = Min(column, false, nothing)

fresh(s::Min) = Min(s.column)
function update!(s::Min, row)
    v = row[s.column]
    s.lo = s.seen ? min(s.lo, v) : v
    s.seen = true
    return nothing
end
value(s::Min) = NamedTuple{(Symbol(s.column, :_min),)}((s.seen ? s.lo : missing,))

"""
    Max(column) -> Summarizer

Tracks the maximum of `column`. Produces the output column
`Symbol(column, :_max)`, e.g. `Max(:x)` produces `:x_max`. The maximum of no
rows is `missing`.
"""
mutable struct Max <: Summarizer
    column::Symbol
    seen::Bool
    hi::Any
end
Max(column::Symbol) = Max(column, false, nothing)

fresh(s::Max) = Max(s.column)
function update!(s::Max, row)
    v = row[s.column]
    s.hi = s.seen ? max(s.hi, v) : v
    s.seen = true
    return nothing
end
value(s::Max) = NamedTuple{(Symbol(s.column, :_max),)}((s.seen ? s.hi : missing,))

"""
    First(column) -> Summarizer

Keeps the value of `column` from the first row folded in. Produces the output
column `Symbol(column, :_first)`, e.g. `First(:x)` produces `:x_first`. The
first of no rows is `missing`.
"""
mutable struct First <: Summarizer
    column::Symbol
    seen::Bool
    val::Any
end
First(column::Symbol) = First(column, false, nothing)

fresh(s::First) = First(s.column)
function update!(s::First, row)
    if !s.seen
        s.val = row[s.column]
        s.seen = true
    end
    return nothing
end
value(s::First) = NamedTuple{(Symbol(s.column, :_first),)}((s.seen ? s.val : missing,))

"""
    Last(column) -> Summarizer

Keeps the value of `column` from the most recent row folded in. Produces the
output column `Symbol(column, :_last)`, e.g. `Last(:x)` produces `:x_last`.
The last of no rows is `missing`.
"""
mutable struct Last <: Summarizer
    column::Symbol
    seen::Bool
    val::Any
end
Last(column::Symbol) = Last(column, false, nothing)

fresh(s::Last) = Last(s.column)
function update!(s::Last, row)
    s.val = row[s.column]
    s.seen = true
    return nothing
end
value(s::Last) = NamedTuple{(Symbol(s.column, :_last),)}((s.seen ? s.val : missing,))

# --- shared plumbing -------------------------------------------------------

tosummarizers(s::Summarizer) = Summarizer[s]
tosummarizers(ss) = collect(Summarizer, ss)

tokeycolumns(::Nothing) = Symbol[]
tokeycolumns(k::Symbol) = Symbol[k]
tokeycolumns(ks) = collect(Symbol, ks)

# Deduplicate prototypes by output-name tuple (identical configurations
# collapse to one shared instance) and validate the surviving output names
# against :time, the key columns, and each other.
function prototypes(ss::Vector{Summarizer}, keycols::Vector{Symbol})
    isempty(ss) && throw(ArgumentError("at least one summarizer is required"))
    protos = Summarizer[]
    seen = Set{Tuple{Vararg{Symbol}}}()
    used = Set{Symbol}()
    for s in ss
        outnames = keys(value(s))
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
    return protos
end

freshall(protos::Vector{Summarizer}) = Summarizer[fresh(s) for s in protos]

summaryvalues(states::Vector{Summarizer}) =
    reduce(merge, (value(s) for s in states); init = (;))

keyvalues(row, keynames::Tuple{Vararg{Symbol}}) =
    NamedTuple{keynames}(Tuple(row[c] for c in keynames))

groupstates!(groups, protos, row, keynames) =
    get!(() -> freshall(protos), groups, keyvalues(row, keynames))

sortedgroups(groups) = sort!(collect(groups); by = kv -> Tuple(first(kv)))

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
            keynames = Tuple(keycols)
            states = freshall(protos)
            groups = Dict{Any,Vector{Summarizer}}()
            step = function (c)
                for r in eachrow(c)
                    rstates = isempty(keycols) ? states :
                        groupstates!(groups, protos, r, keynames)
                    foreach(s -> update!(s, r), rstates)
                end
                return nothing
            end
            flush = function ()
                isempty(keycols) && return DataFrame(NamedTuple[
                    merge((; time = ctx.stop), summaryvalues(states))])
                rows = NamedTuple[merge((; time = ctx.stop), k, summaryvalues(gs))
                                  for (k, gs) in sortedgroups(groups)]
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
            keynames = Tuple(keycols)
            groups = Dict{Any,Vector{Summarizer}}()
            states = Summarizer[]
            started = false
            cycletime = nothing
            function closecycle!(rows)
                if isempty(keycols)
                    push!(rows, merge((; time = cycletime), summaryvalues(states)))
                else
                    for (k, kstates) in sortedgroups(groups)
                        push!(rows, merge((; time = cycletime), k,
                                          summaryvalues(kstates)))
                    end
                    empty!(groups)
                end
                return nothing
            end
            # A cycle closes when a row with a later time arrives (causal), so
            # the open cycle's state is buffered across chunk boundaries and
            # only flushed at the end of the stream.
            step = function (c)
                rows = NamedTuple[]
                for r in eachrow(c)
                    if !started || r.time != cycletime
                        started && closecycle!(rows)
                        cycletime = r.time
                        started = true
                        isempty(keycols) && (states = freshall(protos))
                    end
                    cstates = isempty(keycols) ? states :
                        groupstates!(groups, protos, r, keynames)
                    foreach(s -> update!(s, r), cstates)
                end
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            flush = function ()
                started || return nothing
                rows = NamedTuple[]
                closecycle!(rows)
                return DataFrame(rows)
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
            keynames = Tuple(keycols)
            groups = Dict{Any,Vector{Summarizer}}()
            states = freshall(protos)
            checked = false   # collision check needs the schema: first chunk
            step = function (c)
                if !checked
                    for s in protos, n in keys(value(s))
                        String(n) in names(c) && throw(ArgumentError(
                            "summary column $n collides with an existing column"))
                    end
                    checked = true
                end
                vals = NamedTuple[]
                for r in eachrow(c)
                    rstates = isempty(keycols) ? states :
                        groupstates!(groups, protos, r, keynames)
                    foreach(s -> update!(s, r), rstates)
                    push!(vals, summaryvalues(rstates))
                end
                return hcat(c, DataFrame(vals))
            end
            return chunkmap(step, p.run(ctx))
        end
    end
end
