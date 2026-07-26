# The summarization transforms (summarize, summarizecycles, addsummarycolumns)
# drive summarizers (summarizers.jl) over the stream of chunks, carrying state
# across chunk boundaries. The per-row folding lives in kernels behind a
# function barrier: the type-unstable setup — reading the schema, building or
# widening the states, turning the chunk into a column table — happens once
# per chunk, and the kernels then take concretely typed arguments and
# specialize.

# --- shared plumbing -------------------------------------------------------

tosummarizers(s::Summarizer) = Summarizer[s]
tosummarizers(ss) = collect(Summarizer, ss)

tokeycolumns(::Nothing) = Symbol[]
tokeycolumns(k::Symbol) = Symbol[k]
tokeycolumns(ks) = collect(Symbol, ks)

# Expand the requested summarizers into the full set to fold — each one's
# dependencies recursively, deduplicated by output-name tuple (identical
# configurations collapse to one shared instance) and ordered topologically by
# a post-order depth-first walk, so every state precedes its dependents in the
# tuple and its values are accumulated first at emission time. Output names
# are validated against each other across the whole expanded set, but against
# :time and the key columns only for the *requested* names — a hidden
# dependency never reaches the output, so it cannot collide with it. Returns
# the expanded prototypes as a tuple (so the states derived from them are a
# concrete tuple too and the folding loops specialize on it) plus the
# requested output names, in request order, for projecting emitted rows.
function prototypes(ss::Vector{Summarizer}, keycols::Vector{Symbol})
    isempty(ss) && throw(ArgumentError("at least one summarizer is required"))
    protos = Summarizer[]
    seen = Set{Tuple{Vararg{Symbol}}}()      # finished, by output-name tuple
    visiting = Set{Tuple{Vararg{Symbol}}}()  # walk in progress: cycle guard
    used = Set{Symbol}()
    function expand(s::Summarizer)
        outnames = keys(emptyvalue(s))
        outnames in seen && return
        outnames in visiting && throw(ArgumentError(
            "summarizer dependency cycle through $(first(outnames))"))
        push!(visiting, outnames)
        foreach(expand, dependencies(s))
        delete!(visiting, outnames)
        for n in outnames
            n in used && throw(
                ArgumentError(
                    "output column $n is produced by more than one summarizer"),
            )
            push!(used, n)
        end
        push!(seen, outnames)
        push!(protos, s)
        return
    end
    requested = Symbol[]
    for s in ss
        outnames = keys(emptyvalue(s))
        for n in outnames
            n === :time && throw(ArgumentError(
                "summarizer output column may not be named time"))
            n in keycols && throw(
                ArgumentError(
                    "summarizer output column $n collides with a key column"),
            )
        end
        expand(s)
        for n in outnames
            n in requested || push!(requested, n)
        end
    end
    return Tuple(protos), Tuple(requested)
end

# Input column element types, mirroring the row access in update!.
chunktypes(c::DataFrame) =
    NamedTuple{Tuple(propertynames(c))}(Tuple(eltype(col) for col in eachcol(c)))

# A source may infer a column's element type per chunk, so the state types
# track the promotion of every input type seen so far rather than trusting the
# first chunk.
promotetypes(::Nothing, b::NamedTuple) = b
promotetypes(a::NamedTuple, b::NamedTuple) =
    NamedTuple{keys(b)}(
        Tuple(haskey(a, k) ? promote_type(a[k], b[k]) : b[k]
              for k in keys(b)),
    )

newstates(protos::Tuple, intypes::NamedTuple) = map(s -> fresh(s, intypes), protos)
widenstates(states::Tuple, intypes::NamedTuple) =
    map(st -> widenstate(st, intypes), states)

# The per-key state tuples of the transforms that summarize by key, plus the
# two buffers that make *closing* a group of them free. Both type parameters
# are concrete — the key type comes from the first row, the value type from the
# state prototypes — so the folding kernels specialize on this rather than on
# the Dict{Any,Vector{Summarizer}} it would otherwise be.
#
# `summarize` and `addsummarycolumns` never close their table and leave the two
# buffers empty. The cycle-closing transforms (`summarizecycles`,
# `intervalize`) close one per timestamp or per interval, and there the buffers
# are the difference between an allocation-free fold and one that rebuilds a
# Dict entry, a state tuple per key, and a sort buffer every cycle.
mutable struct GroupTable{K,S<:Tuple}
    table::Dict{K,S}
    scratch::Vector{Pair{K,S}}  # key-ordered emission buffer, reused
    pool::Vector{S}             # retired state tuples, zeroed on reuse
end

GroupTable{K,S}() where {K,S<:Tuple} =
    GroupTable{K,S}(Dict{K,S}(), Pair{K,S}[], S[])

Base.keytype(::GroupTable{K}) where {K} = K
Base.valtype(::GroupTable{K,S}) where {K,S} = S

# `stateprotos` must already be widened: it is what fixes the rebuilt table's
# value type, which a comprehension over an empty table could not. The pool is
# deliberately not carried over — a retired tuple has the pre-widening type.
function widengroups(gt::GroupTable{K}, stateprotos::S,
    intypes::NamedTuple) where {K,S}
    widened = GroupTable{K,S}()
    for (k, gs) in gt.table
        widened.table[k] = widenstates(gs, intypes)
    end
    return widened
end

newgroups(stateprotos::S, nt::NamedTuple, keynames::Val) where {S} =
    GroupTable{typeof(keyvalues(first(Tables.rows(nt)), keynames)),S}()

# This key's state tuple, recycling a retired one where there is one. Zeroing
# on the way *out* rather than on retirement keeps the cost proportional to the
# tuples actually reused. The key is left unconstrained so the Dict converts it,
# as it did before: a lookup may carry narrower value types than the table's
# key type (asofjoin's store makes the same allowance).
@inline groupstates!(gt::GroupTable{K,S}, key, stateprotos::S) where {K,S<:Tuple} =
    get!(gt.table, key) do
        isempty(gt.pool) ? map(fresh, stateprotos) : freshall!(pop!(gt.pool))
    end

# Values accumulate left to right over the topologically ordered state tuple,
# each state seeing the values of everything before it — which is how a
# dependent summarizer reads its dependencies. The states peel off as
# positional arguments (afoldl-style) rather than by Base.tail on a tuple:
# the accumulated NamedTuple grows while the states shrink, and only the
# vararg form keeps inference from widening on that recursion. The
# accumulated values are then projected down to the requested output names,
# riding in a Val like the key names, so hidden dependencies are folded but
# never emitted.
@inline accvalues(vals::NamedTuple) = vals
@inline accvalues(vals::NamedTuple, st, rest...) =
    accvalues(merge(vals, value(st, vals)), rest...)
@inline summaryvalues(states::Tuple, ::Val{R}) where {R} =
    NamedTuple{R}(accvalues((;), states...))
emptyvalues(protos::Tuple, ::Val{R}) where {R} =
    NamedTuple{R}(merge(map(emptyvalue, protos)...))

@inline summaryrow(t, states::Tuple, r::Val) =
    merge((; time = t), summaryvalues(states, r))
@inline summaryrow(t, k::NamedTuple, states::Tuple, r::Val) =
    merge((; time = t), k, summaryvalues(states, r))

# The key names ride in a Val so the group key's NamedTuple type — and hence
# the group table's Dict type — is known to the compiler.
@inline keyvalues(row, ::Val{KN}) where {KN} =
    NamedTuple{KN}(map(c -> getproperty(row, c), KN))

# Key order is the emission order everywhere a group table is drained. The key
# is a NamedTuple, so `values` is already the tuple to compare on — tuples
# order lexicographically, which is the documented "sorted by key value".
@inline groupkey(kv::Pair) = values(first(kv))

# For the once-per-run drains (`summarize`'s flush), where a fresh vector costs
# nothing; the per-cycle drain uses the reusable buffer in `closecycle!`.
sortedgroups(gt::GroupTable) = sort!(collect(gt.table); by = groupkey)

# The row types the kernels emit. The state prototypes cannot simply be run
# through `value` to find out: Min/Max/First/Last leave their value field
# undefined until a row is folded in.
rowtype(::Type{T}, ::Type{S}, ::Val{R}) where {T,S,R} =
    Base.promote_op(summaryrow, T, S, Val{R})
rowtype(::Type{T}, ::Type{K}, ::Type{S}, ::Val{R}) where {T,K,S,R} =
    Base.promote_op(summaryrow, T, K, S, Val{R})
valuetype(::Type{S}, ::Val{R}) where {S,R} =
    Base.promote_op(summaryvalues, S, Val{R})

# The output value type an empty-tolerant transform emits: the summary values'
# NamedTuple type (from the realized states `S`) promoted field-wise with the
# empty values' (an empty window or interval emits the latter), so e.g. `Min`
# over an `Int` column gives `Union{Missing, Int}`. `promote_op` can in
# principle fail to concretize, hence the `Union` fallback. Shared by
# `addrollingcolumns` (rolling.jl) and `intervalize` (intervalize.jl).
function promotedvaluetype(::Type{S}, protos::Tuple, outs::Val) where {S}
    VT = valuetype(S, outs)
    e = emptyvalues(protos, outs)
    E = typeof(e)
    (VT <: NamedTuple && isconcretetype(VT) && fieldnames(VT) == keys(e)) ||
        return Union{VT,E}
    return NamedTuple{keys(e),
        Tuple{
            ntuple(i -> promote_type(fieldtype(VT, i), fieldtype(E, i)),
                fieldcount(E))...,
        }}
end

# Per-run mutable state shared by the three transforms. It lives in fields
# rather than in the step/flush closures' captured locals because captured
# variables that are reassigned get boxed. The dynamically typed fields are
# by design: the state types depend on the first chunk's schema, and
# everything reading them per row sits behind the folding kernels' function
# barrier. Each transform uses the subset of fields it needs.
mutable struct SummaryFold
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    widened::Bool          # whether the last chunk moved that promotion
    stateprotos::Any       # state tuple serving as the template for fresh copies
    states::Any            # keyless transforms: the running state tuple
    groups::Any            # keyed transforms: Dict of key => state tuple
    cycletime::Any         # summarizecycles: the open cycle's time
    checked::Bool          # addsummarycolumns: collision check done
end
SummaryFold() = SummaryFold(nothing, false, nothing, nothing, nothing, nothing,
    false)

# Per-chunk setup shared by the transforms: promote the schema, then build the
# states on the first chunk or widen them when the promotion has moved.
# Returns the chunk as a column table for the kernels.
function preparechunk!(fold::SummaryFold, protos::Tuple, keyed::Bool,
    keynames::Val, c::DataFrame)
    types = promotetypes(fold.types, chunktypes(c))
    fold.widened = fold.types !== nothing && types != fold.types
    fold.types = types
    nt = Tables.columntable(c)
    if fold.stateprotos === nothing
        fold.stateprotos = newstates(protos, types)
        keyed ? (fold.groups = newgroups(fold.stateprotos, nt, keynames)) :
        (fold.states = map(fresh, fold.stateprotos))
    elseif fold.widened
        fold.stateprotos = widenstates(fold.stateprotos, types)
        keyed ? (fold.groups = widengroups(fold.groups, fold.stateprotos, types)) :
        (fold.states = widenstates(fold.states, types))
    end
    return nt
end

# --- folding kernels -------------------------------------------------------
#
# Everything below is called once per chunk with concretely typed arguments,
# so each specializes on the state tuple and the chunk's column table and the
# per-row work compiles down to direct field access.

function foldall!(states::Tuple, nt::NamedTuple)
    for row in Tables.rows(nt)
        updateall!(states, row)
    end
    return nothing
end

function foldgroups!(gt::GroupTable{K,S}, stateprotos::S, nt::NamedTuple,
    ::Val{KN}) where {K,S,KN}
    for row in Tables.rows(nt)
        states = groupstates!(gt, keyvalues(row, Val(KN)), stateprotos)
        updateall!(states, row)
    end
    return nothing
end

# A cycle closes when a row with a later time arrives (causal), so the open
# cycle's state is carried across chunk boundaries and only closed by flush.
function foldcycles!(states::S, nt::NamedTuple, cycletime,
    r::Val) where {S<:Tuple}
    rows = rowtype(eltype(nt.time), S, r)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            # summaryrow has already copied the values out, so the closed
            # cycle's states can be zeroed and reused rather than replaced.
            cycletime === nothing ||
                push!(rows, summaryrow(something(cycletime), states, r))
            cycletime = t
            states = freshall!(states)
        end
        updateall!(states, row)
    end
    return rows, states, cycletime
end

function foldcyclesgrouped!(gt::GroupTable{K,S}, stateprotos::S, nt::NamedTuple,
    cycletime, ::Val{KN}, r::Val) where {K,S,KN}
    rows = rowtype(eltype(nt.time), K, S, r)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            cycletime === nothing ||
                closecycle!(rows, gt, something(cycletime), r)
            cycletime = t
        end
        states = groupstates!(gt, keyvalues(row, Val(KN)), stateprotos)
        updateall!(states, row)
    end
    return rows, cycletime
end

# Emit one row per present key, in key order, then retire the state tuples to
# the pool and clear the table for the next cycle. `summaryrow` has copied the
# values out by then, so a retired tuple carries nothing that is still needed —
# `groupstates!` zeroes it before it is handed out again. Both buffers are
# reused, so a close allocates only the rows it pushes.
function closecycle!(rows, gt::GroupTable, t, r::Val)
    scratch = gt.scratch
    empty!(scratch)
    append!(scratch, gt.table)
    sort!(scratch; by = groupkey)
    for (k, states) in scratch
        push!(rows, summaryrow(t, k, states, r))
        push!(gt.pool, states)
    end
    empty!(gt.table)
    return rows
end

function foldrunning!(states::S, nt::NamedTuple, n::Int,
    r::Val) where {S<:Tuple}
    vals = Vector{valuetype(S, r)}(undef, n)
    i = 0
    for row in Tables.rows(nt)
        updateall!(states, row)
        vals[i+=1] = summaryvalues(states, r)
    end
    return vals
end

function foldrunninggrouped!(gt::GroupTable{K,S}, stateprotos::S, nt::NamedTuple,
    n::Int, ::Val{KN}, r::Val) where {K,S,KN}
    vals = Vector{valuetype(S, r)}(undef, n)
    i = 0
    for row in Tables.rows(nt)
        states = groupstates!(gt, keyvalues(row, Val(KN)), stateprotos)
        updateall!(states, row)
        vals[i+=1] = summaryvalues(states, r)
    end
    return vals
end

# --- summarization transforms ----------------------------------------------

"""
    summarize(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)
    summarize(p::CausalPipeline, summarizers; key = nothing) -> CausalPipeline

A transform summarizing the whole context: every input row is folded into the
summarizers and a single batch of rows is emitted at the context's end time
`stop`, dropping the input columns. `summarizers` is a [`Summarizer`](@ref)
or a collection of them; the output columns are `time`, the key columns, then
each summarizer's value columns.

Without `key` the output is exactly one row (the identity summary — e.g.
`count = 0` — when the input is empty). With `key` (a column name or
collection of column names) one row is emitted per unique key value, sorted
by key; an empty input yields no rows.

The curried form composes with `|>`; the uncurried form applies directly, so
`summarize(p, ss; key)` is equivalent to `p |> summarize(ss; key)`.
"""
function summarize(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            outs = Val(requested)
            keyed = !isempty(keycols)
            fold = SummaryFold()
            step = function (c)
                nt = preparechunk!(fold, protos, keyed, keynames, c)
                keyed ? foldgroups!(fold.groups, fold.stateprotos, nt, keynames) :
                foldall!(fold.states, nt)
                return nothing
            end
            flush = function ()
                if !keyed
                    vals =
                        fold.states === nothing ? emptyvalues(protos, outs) :
                        summaryvalues(fold.states, outs)
                    return DataFrame([merge((; time = ctx.stop), vals)])
                end
                fold.groups === nothing && return nothing
                rows = [
                    summaryrow(ctx.stop, k, gs, outs)
                    for (k, gs) in sortedgroups(fold.groups)
                ]
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            return chunkmap(step, p.run(ctx); flush = flush)
        end
    end
end
summarize(p::CausalPipeline, summarizers; kwargs...) =
    summarize(summarizers; kwargs...)(p)

"""
    summarizecycles(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizecycles(p::CausalPipeline, summarizers; key = nothing) -> CausalPipeline

A transform summarizing each *cycle* — a maximal run of rows sharing one
timestamp — independently, with fresh state per cycle. For every cycle one
row is emitted at the cycle's time (per unique key value, sorted by key, when
`key` is given), dropping the input columns. A cycle spanning a chunk
boundary is summarized as a single cycle.

The curried form composes with `|>`; the uncurried form applies directly, so
`summarizecycles(p, ss; key)` is equivalent to `p |> summarizecycles(ss; key)`.
"""
function summarizecycles(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            outs = Val(requested)
            keyed = !isempty(keycols)
            fold = SummaryFold()
            step = function (c)
                nt = preparechunk!(fold, protos, keyed, keynames, c)
                rows = if keyed
                    rs, fold.cycletime = foldcyclesgrouped!(
                        fold.groups, fold.stateprotos, nt, fold.cycletime,
                        keynames, outs)
                    rs
                else
                    rs, fold.states, fold.cycletime = foldcycles!(
                        fold.states, nt, fold.cycletime, outs)
                    rs
                end
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            flush = function ()
                fold.cycletime === nothing && return nothing
                rows =
                    keyed ?
                    closecycle!(
                        rowtype(typeof(fold.cycletime), keytype(fold.groups),
                            valtype(fold.groups), outs)[], fold.groups,
                        fold.cycletime, outs) :
                    [summaryrow(fold.cycletime, fold.states, outs)]
                return isempty(rows) ? nothing : DataFrame(rows)
            end
            return chunkmap(step, p.run(ctx); flush = flush)
        end
    end
end
summarizecycles(p::CausalPipeline, summarizers; kwargs...) =
    summarizecycles(summarizers; kwargs...)(p)

"""
    addsummarycolumns(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)
    addsummarycolumns(p::CausalPipeline, summarizers; key = nothing) -> CausalPipeline

A transform keeping all existing columns and appending each summarizer's
value columns, holding the running summary *after* that row has been folded
in. With `key` (a column name or collection of column names) a separate
running summary is kept per unique key value. State runs over the whole
window, carried across chunk boundaries. The output columns may not collide
with existing columns.

The curried form composes with `|>`; the uncurried form applies directly, so
`addsummarycolumns(p, ss; key)` is equivalent to
`p |> addsummarycolumns(ss; key)`.
"""
function addsummarycolumns(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested = prototypes(tosummarizers(summarizers), keycols)
            keynames = Val(Tuple(keycols))
            outs = Val(requested)
            keyed = !isempty(keycols)
            fold = SummaryFold()
            step = function (c)
                if !fold.checked   # needs the schema: first chunk only
                    for n in requested   # hidden dependencies are never added
                        String(n) in names(c) && throw(
                            ArgumentError(
                                "summary column $n collides with an existing column",
                            ),
                        )
                    end
                    fold.checked = true
                end
                nt = preparechunk!(fold, protos, keyed, keynames, c)
                vals =
                    keyed ?
                    foldrunninggrouped!(fold.groups, fold.stateprotos,
                        nt, nrow(c), keynames, outs) :
                    foldrunning!(fold.states, nt, nrow(c), outs)
                return hcat(c, DataFrame(vals); copycols = false)
            end
            return chunkmap(step, p.run(ctx))
        end
    end
end
addsummarycolumns(p::CausalPipeline, summarizers; kwargs...) =
    addsummarycolumns(summarizers; kwargs...)(p)
