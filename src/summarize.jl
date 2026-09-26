# The summarization transforms (summarize, summarizecycles, addsummarycolumns)
# drive summarizers over the stream, carrying state across chunks. Type-unstable
# setup (reading the schema, building or widening the states, making the column
# table) runs once per chunk; the per-row folding runs in kernels behind a
# function barrier, over concretely typed arguments.

# --- shared plumbing -------------------------------------------------------

tosummarizers(s::Summarizer) = Summarizer[s]
tosummarizers(ss) = collect(Summarizer, ss)

tokeycolumns(::Nothing) = Symbol[]
tokeycolumns(k::Symbol) = Symbol[k]
# A string is one name, not a collection of characters.
tokeycolumns(k::AbstractString) = Symbol[Symbol(k)]
tokeycolumns(ks) = collect(Symbol, ks)

# A keyed transform's `key`, normalized and checked eagerly: the time column
# orders the stream, so it is never a key.
function keycolumns(key, op::String)
    keycols = tokeycolumns(key)
    allunique(keycols) || throw(ArgumentError("$op key columns must be unique"))
    :time in keycols && throw(ArgumentError("$op key may not be :time"))
    return keycols
end

# Called on the first chunk, so a missing key column isn't a getproperty error
# deep in a kernel. `input` names the stream, for binary transforms.
function checkkeycolumns(keycols::Vector{Symbol}, c::DataFrame, op::String,
    input::String = "the input")
    for k in keycols
        String(k) in names(c) ||
            throw(ArgumentError("$op key column $(repr(k)) not found in $input"))
    end
    return nothing
end

# A declared key set (`keyset`), which makes a keyed transform dense: every
# close emits one row per declared key, in declared order. The keys are built
# at construction as NamedTuples over the key columns, so the declaration, not
# the data, fixes the output key type. A data key of another `isequal` type (an
# `Int` against a declared `Float64`) is looked up without conversion. `op`
# names the transform in errors.
struct KeySet{K<:NamedTuple}
    keys::Vector{K}
    index::Dict{K,Int}
    op::String
end

tokeyset(::Nothing, keycols::Vector{Symbol}, op::String) = nothing
function tokeyset(keyset, keycols::Vector{Symbol}, op::String)
    isempty(keycols) && throw(ArgumentError("$op keyset requires a key"))
    KN = Tuple(keycols)
    tuples = map(v -> keysettuple(v, KN, op), collect(keyset))
    # `map` narrows each column to the typejoin of its values, so an ordinary
    # declaration gives a concrete key type
    cols = ntuple(j -> map(t -> t[j], tuples), length(KN))
    K = NamedTuple{KN,Tuple{map(eltype, cols)...}}
    keys = K[K(map(col -> col[i], cols)) for i in eachindex(tuples)]
    allunique(keys) || throw(ArgumentError("$op keyset values must be unique"))
    return KeySet{K}(keys, Dict{K,Int}(k => i for (i, k) in enumerate(keys)), op)
end

# One declared key as a tuple of column values. It is declared as the value
# itself for a single key column, otherwise as a tuple of values or a
# NamedTuple naming exactly the key columns.
function keysettuple(v, KN::Tuple{Vararg{Symbol}}, op::String)
    length(KN) == 1 && return (v,)
    if v isa NamedTuple
        (length(v) == length(KN) && all(in(keys(v)), KN)) &&
            return Tuple(NamedTuple{KN}(v))
    elseif v isa Tuple
        length(v) == length(KN) && return v
    end
    throw(
        ArgumentError(
            "$op keyset element $(repr(v)) must be a tuple or named tuple of " *
            "values for the key columns $(join(map(repr, KN), ", "))"),
    )
end

# The declared slot of a row's key, or an error for a key outside the set.
@inline function keyindex(ks::KeySet, k)
    i = get(ks.index, k, 0)
    i == 0 && throwundeclared(ks, k)
    return i
end
@noinline throwundeclared(ks::KeySet, k) =
    throw(ArgumentError("$(ks.op) key $k is not in the declared keyset"))

# Membership only, for the window kernels, whose groups are not slot-indexed.
# The `Nothing` method (no declared keys) compiles away.
@inline checkdeclared(::Nothing, row, keynames::Val) = nothing
@inline checkdeclared(ks::KeySet, row, keynames::Val) =
    (keyindex(ks, keyvalues(row, keynames)); nothing)

# Expand the requested summarizers into the full set to fold: dependencies
# recursively, deduplicated by output-name tuple and ordered topologically
# (post-order depth-first), so every state precedes its dependents. Output
# names must be unique across the whole set, but only the *requested* names
# are checked against :time and the key columns, since hidden dependencies
# never reach the output. Returns the prototypes as a tuple (so the states are
# a concrete tuple) and the requested output names, in request order.
function prototypes(
    ss::Vector{Summarizer},
    keycols::Vector{Symbol},
    op::String = "summarize",
)
    isempty(ss) && throw(ArgumentError("$op requires at least one summarizer"))
    protos = Summarizer[]
    # A linear `in` over the few output-name tuples beats a Set keyed by an
    # abstract type.
    seen = Tuple{Vararg{Symbol}}[]      # finished, by output-name tuple
    visiting = Tuple{Vararg{Symbol}}[]  # walk in progress: cycle guard
    used = Set{Symbol}()
    function expand(s::Summarizer)
        outnames = keys(emptyvalue(s))
        outnames in seen && return
        outnames in visiting && throw(
            ArgumentError(
                "$op: summarizer dependency cycle through $(repr(first(outnames)))"),
        )
        push!(visiting, outnames)          # a stack: the walk is depth-first
        foreach(expand, dependencies(s))
        pop!(visiting)
        for n in outnames
            n in used && throw(
                ArgumentError(
                    "$op output column $(repr(n)) is produced by more than one summarizer",
                ),
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
                "$op output column may not be named :time"))
            n in keycols && throw(
                ArgumentError(
                    "$op output column $(repr(n)) collides with a key column"),
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

# A column's element type may differ between chunks, so the state types track
# the promotion of every input type seen.
promotetypes(::Nothing, b::NamedTuple) = b
promotetypes(a::NamedTuple, b::NamedTuple) =
    NamedTuple{keys(b)}(
        Tuple(haskey(a, k) ? promote_type(a[k], b[k]) : b[k]
              for k in keys(b)),
    )

newstates(protos::Tuple, intypes::NamedTuple) = map(s -> fresh(s, intypes), protos)
widenstates(states::Tuple, intypes::NamedTuple) =
    map(st -> widenstate(st, intypes), states)

# The per-key state tuples of the keyed transforms, plus two buffers that make
# closing the table allocation-free. Both type parameters are concrete (the key
# type from the first row, the state type from the prototypes), so the kernels
# specialize on them. Only the cycle-closing transforms (`summarizecycles`,
# `intervalize`) close tables and use the buffers.
mutable struct GroupTable{K,S<:Tuple}
    table::Dict{K,S}
    scratch::Vector{Pair{K,S}}  # key-ordered emission buffer, reused
    pool::Vector{S}             # retired state tuples, zeroed on reuse
end

GroupTable{K,S}() where {K,S<:Tuple} =
    GroupTable{K,S}(Dict{K,S}(), Pair{K,S}[], S[])

Base.keytype(::GroupTable{K}) where {K} = K
Base.valtype(::GroupTable{K,S}) where {K,S} = S

# `stateprotos` must already be widened: it fixes the new table's value type,
# which an empty table couldn't. The pool is dropped, since retired tuples have
# the old type.
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

# This key's state tuple, recycling a retired one if available. Zeroing on the
# way *out* costs only for tuples actually reused. The key is unconstrained so
# the Dict converts it: a row's key may have narrower types than the table's.
@inline groupstates!(gt::GroupTable{K,S}, key, stateprotos::S) where {K,S<:Tuple} =
    get!(gt.table, key) do
        isempty(gt.pool) ? map(fresh, stateprotos) : freshall!(pop!(gt.pool))
    end

# Values accumulate left to right over the topologically ordered states, each
# seeing the values of those before it, which is how a dependent reads its
# dependencies. The states peel off as varargs rather than by Base.tail, which
# keeps inference from widening as the NamedTuple grows. The result is then
# projected onto the requested names (in a Val), dropping hidden dependencies.
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

# The key names ride in a Val so the key's NamedTuple type is known statically.
@inline keyvalues(row, ::Val{KN}) where {KN} =
    NamedTuple{KN}(map(c -> getproperty(row, c), KN))

# Groups are emitted in key order: the key's values as a tuple, compared
# lexicographically.
@inline groupkey(kv::Pair) = values(first(kv))

# For the once-per-run drains (`summarize`'s and `lastrow`'s flush); the
# per-cycle drain reuses `closecycle!`'s buffer. `lastrow` passes a bare Dict.
sortedgroups(d::AbstractDict) = sort!(collect(d); by = groupkey)
sortedgroups(gt::GroupTable) = sortedgroups(gt.table)

# The row types the kernels emit, by inference: calling `value` on an unfolded
# state won't do, since Min/Max/First/Last leave their value undefined.
rowtype(::Type{T}, ::Type{S}, ::Val{R}) where {T,S,R} =
    Base.promote_op(summaryrow, T, S, Val{R})
rowtype(::Type{T}, ::Type{K}, ::Type{S}, ::Val{R}) where {T,K,S,R} =
    Base.promote_op(summaryrow, T, K, S, Val{R})
valuetype(::Type{S}, ::Val{R}) where {S,R} =
    Base.promote_op(summaryvalues, S, Val{R})

# The value type of a transform that also emits empty values: the summary
# values' type (from the states `S`) promoted field-wise with the empty values',
# so `Min` over `Int` gives `Union{Missing, Int}`. The `Union` fallback covers
# a `promote_op` that fails to concretize.
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

# A keyed grid row (`:time`, the key, then the values) and its type, for paths
# that emit empty values beside summaries.
gridrow(t, k, v) = merge((; time = t), k, v)
gridrowtype(::Type{T}, ::Type{K}, ::Type{V}) where {T,K,V} =
    Base.promote_op(gridrow, T, K, V)

# --- dense (declared-key) groups -------------------------------------------
#
# The per-key state tuples of a dense transform, one slot per declared key in
# `KeySet` order, built up front. A row costs one Dict lookup, and a close is a
# walk over the slots with no sort or pool. `folded` marks the slots that took
# a row since the last close; only those need zeroing.
struct DenseGroups{S<:Tuple}
    states::Vector{S}
    folded::Vector{Bool}
end

densegroups(stateprotos::S, n::Int) where {S<:Tuple} =
    DenseGroups{S}(S[map(fresh, stateprotos) for _ in 1:n], fill(false, n))

# `stateprotos` must already be widened, as in `widengroups`.
widendense(dg::DenseGroups, stateprotos::S, intypes::NamedTuple) where {S} =
    DenseGroups{S}(S[widenstates(gs, intypes) for gs in dg.states], dg.folded)

# The dense row type and empty row. Type-unstable setup, once per chunk.
function densetypes(::Type{T}, stateprotos::S, protos::Tuple, ::KeySet{K},
    outs::Val) where {T,S,K}
    V = promotedvaluetype(S, protos, outs)
    return gridrowtype(T, K, V), convert(V, emptyvalues(protos, outs))
end

@inline function densefold!(dg::DenseGroups, ks::KeySet, row, keynames::Val)
    i = keyindex(ks, keyvalues(row, keynames))
    updateall!(@inbounds(dg.states[i]), row)
    @inbounds dg.folded[i] = true
    return nothing
end

# Emit one row per declared key at `t` (the summary for a slot that folded
# rows, else the empty values), zeroing the folded slots once their values are
# copied out. Allocates only the rows it pushes.
function closedense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet, t, r::Val,
    emptyrow) where {RT}
    states, folded = dg.states, dg.folded
    for i in eachindex(ks.keys, states, folded)
        k = @inbounds ks.keys[i]
        if @inbounds folded[i]
            gs = @inbounds states[i]
            push!(rows, convert(RT, gridrow(t, k, summaryvalues(gs, r))))
            @inbounds states[i] = freshall!(gs)
            @inbounds folded[i] = false
        else
            push!(rows, convert(RT, gridrow(t, k, emptyrow)))
        end
    end
    return rows
end

# Per-run state shared by the summarization transforms (and intervalize), in
# fields because reassigned closure captures are boxed. The state types depend
# on the schema, so these fields are untyped; per-row work sits behind the
# kernels' function barrier. Each transform uses the fields it needs.
mutable struct SummaryFold
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    widened::Bool          # whether the last chunk moved that promotion
    stateprotos::Any       # state tuple serving as the template for fresh copies
    states::Any            # keyless transforms: the running state tuple
    groups::Any            # keyed transforms: a GroupTable or DenseGroups
    cycletime::Any         # summarizecycles: the open cycle's time
    checked::Bool          # addsummarycolumns: collision check done
end
SummaryFold() = SummaryFold(nothing, false, nothing, nothing, nothing, nothing,
    false)

# Per-chunk setup: promote the schema, then build the states on the first
# chunk or widen them when the promotion changes. A declared `keyset` uses
# `DenseGroups` rather than a `GroupTable`. Returns the column table.
function preparechunk!(fold::SummaryFold, protos::Tuple, keyed::Bool,
    keynames::Val, c::DataFrame; keyset::Union{Nothing,KeySet} = nothing)
    types = promotetypes(fold.types, chunktypes(c))
    fold.widened = fold.types !== nothing && types != fold.types
    fold.types = types
    nt = Tables.columntable(c)
    if fold.stateprotos === nothing
        fold.stateprotos = newstates(protos, types)
        if keyset !== nothing
            fold.groups = densegroups(fold.stateprotos, length(keyset.keys))
        elseif keyed
            fold.groups = newgroups(fold.stateprotos, nt, keynames)
        else
            fold.states = map(fresh, fold.stateprotos)
        end
    elseif fold.widened
        fold.stateprotos = widenstates(fold.stateprotos, types)
        if keyset !== nothing
            fold.groups = widendense(fold.groups, fold.stateprotos, types)
        elseif keyed
            fold.groups = widengroups(fold.groups, fold.stateprotos, types)
        else
            fold.states = widenstates(fold.states, types)
        end
    end
    return nt
end

# --- folding kernels -------------------------------------------------------
#
# Called once per chunk with concretely typed arguments, so each specializes on
# the state tuple and column table.

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

# A cycle closes when a later time arrives, so the open cycle carries across
# chunks and the last is closed by flush.
function foldcycles!(states::S, nt::NamedTuple, cycletime,
    r::Val) where {S<:Tuple}
    rows = rowtype(eltype(nt.time), S, r)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            # summaryrow copied the values out, so the states are zeroed and
            # reused.
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
# the pool (`groupstates!` zeroes them on reuse) and clear the table. Allocates
# only the rows it pushes.
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

# The declared-key cycle fold: `foldcyclesgrouped!` over dense slots, so every
# cycle closes with one row per declared key. Pushes into the caller's `rows`
# and returns the open cycle's time.
function foldcyclesdense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet,
    nt::NamedTuple, cycletime, keynames::Val, r::Val, emptyrow) where {RT}
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            cycletime === nothing ||
                closedense!(rows, dg, ks, something(cycletime), r, emptyrow)
            cycletime = t
        end
        densefold!(dg, ks, row, keynames)
    end
    return cycletime
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

A transform summarizing the whole window, emitted at `stop`, with columns
`time`, the key columns, then the summaries; the input columns are dropped.

# Arguments
- `summarizers`: a [`Summarizer`](@ref) or a collection of them. Output names
  must be unique and may not be `time` or a key column.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key the output is one row, the summarizers' empty
  values for an empty input. With a key it is one row per key, sorted by key,
  and nothing for an empty input.
"""
function summarize(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested = prototypes(tosummarizers(summarizers), keycols, "summarize")
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
    summarizecycles(summarizers; key = nothing,
                    keyset = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizecycles(p::CausalPipeline, summarizers; ...) -> CausalPipeline

A transform summarizing each *cycle* — a run of rows sharing one time —
separately, emitting at the cycle's time with columns `time`, the key columns,
then the summaries; the input columns are dropped.

# Arguments
- `summarizers`: a [`Summarizer`](@ref) or a collection of them.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, each cycle emits one row per key present, sorted by
  key.
- `keyset = nothing`: the key values, declared up front (requires `key`), as for
  [`intervalize`](@ref). Every cycle then emits one row per declared key, in
  declared order, with empty values (`count = 0`, `mean = missing`) for keys
  without rows.
"""
function summarizecycles(summarizers; key = nothing, keyset = nothing)
    ks = tokeyset(keyset, tokeycolumns(key), "summarizecycles")
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested =
                prototypes(tosummarizers(summarizers), keycols, "summarizecycles")
            keynames = Val(Tuple(keycols))
            outs = Val(requested)
            keyed = !isempty(keycols)
            fold = SummaryFold()
            step = function (c)
                nt = preparechunk!(fold, protos, keyed, keynames, c; keyset = ks)
                rows = if ks !== nothing
                    RT, emptyrow = densetypes(eltype(nt.time), fold.stateprotos,
                        protos, ks, outs)
                    rs = RT[]
                    fold.cycletime = foldcyclesdense!(rs, fold.groups, ks, nt,
                        fold.cycletime, keynames, outs, emptyrow)
                    rs
                elseif keyed
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
                if ks !== nothing
                    RT, emptyrow = densetypes(typeof(fold.cycletime),
                        fold.stateprotos, protos, ks, outs)
                    drows = closedense!(RT[], fold.groups, ks, fold.cycletime,
                        outs, emptyrow)
                    return isempty(drows) ? nothing : DataFrame(drows)
                end
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

A transform appending running summaries: each row gets the summary of every row
so far, itself included.

# Arguments
- `summarizers`: a [`Summarizer`](@ref) or a collection of them. Their output
  columns may not collide with existing columns.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, each key keeps its own running summary, so a keyed
  [`Count`](@ref) numbers the rows of each key.
"""
function addsummarycolumns(summarizers; key = nothing)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            keycols = tokeycolumns(key)
            protos, requested =
                prototypes(tosummarizers(summarizers), keycols, "addsummarycolumns")
            keynames = Val(Tuple(keycols))
            outs = Val(requested)
            keyed = !isempty(keycols)
            fold = SummaryFold()
            step = function (c)
                if !fold.checked   # needs the schema: first chunk only
                    for n in requested   # hidden dependencies are never added
                        String(n) in names(c) && throw(
                            ArgumentError(
                                "addsummarycolumns output column $(repr(n)) collides with an existing column",
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
