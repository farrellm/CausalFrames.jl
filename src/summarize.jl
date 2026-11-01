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

# The appending transforms' check that none of their output columns `outnames`
# is already a column of the chunk `c`.
function checknewcolumns(c::DataFrame, outnames, op::String)
    for n in outnames
        columnindex(c, n) > 0 && throw(
            ArgumentError(
                "$op output column $(repr(n)) collides with an existing column"),
        )
    end
    return nothing
end

# A declared key set (`keyset`), which makes a keyed transform dense: every
# close emits one row per declared key, in declared order. The keys are built
# at construction as NamedTuples over the key columns, so the declaration, not
# the data, fixes the output key type. A data key of another `isequal` type (an
# `Int` against a declared `Float64`) is looked up without conversion. `op`
# names the transform in errors. Mutable, so the per-chunk dynamic calls that
# take it pass a reference instead of boxing a copy.
mutable struct KeySet{K<:NamedTuple}
    const keys::Vector{K}
    const index::KeyDict{K,Int}
    const op::String
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
    return KeySet{K}(keys, KeyDict{K,Int}(k => i for (i, k) in enumerate(keys)), op)
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

# A keyless grid (`intervalize`, `summarizewindows`) is the dense output over
# one declared key, the empty one, so it shares the declared-key paths. Every
# row has that key, so its lookups compile away.
const NoKey = NamedTuple{(),Tuple{}}
nokeyset(op::String) = KeySet{NoKey}([(;)], KeyDict{NoKey,Int}(((;) => 1,)), op)

# The declared slot of a row's key, or an error for a key outside the set.
@inline function keyindex(ks::KeySet, k)
    i = get(ks.index, k, 0)
    i == 0 && throwundeclared(ks, k)
    return i
end
@inline keyindex(::KeySet{NoKey}, ::NoKey) = 1
@noinline throwundeclared(ks::KeySet, k) =
    throw(ArgumentError("$(ks.op) key $k is not in the declared keyset"))

# Membership only, for the window kernels, whose groups are not slot-indexed.
# The keyset to check on admission is `nothing` when no check is needed, so the
# check compiles away.
@inline checkdeclared(::Nothing, row, keynames::Val) = nothing
@inline checkdeclared(ks::KeySet, row, keynames::Val) =
    (keyindex(ks, keyvalues(row, keynames)); nothing)
admissionkeys(ks::Union{Nothing,KeySet}) = ks
admissionkeys(::KeySet{NoKey}) = nothing

# Expand the requested summarizers into the full set to fold: dependencies
# recursively, deduplicated by output-name tuple and ordered topologically
# (post-order depth-first), so every state precedes its dependents. Output
# names must be unique across the whole set, but only the *requested* names
# are checked against :time and the key columns, since hidden dependencies
# never reach the output. Row terms (`Termed`, or a summarizer's own
# `rowterms`) are unwrapped here, at any depth, and collected by name. Returns
# the prototypes as a tuple (so the states are a concrete tuple), the requested
# output names, in request order, and the row terms as a NamedTuple of
# functions.
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
    terms = Pair{Symbol,Any}[]
    # Unspecialized: it runs once per summarizer, at construction.
    function expand(@nospecialize(t::Summarizer))
        s = unterm(t)
        # the constructor's terms, then the summarizer's own hook (the same
        # terms again when `t` is unwrapped, which `addterm!` skips)
        for x in (t, s), (n, f) in pairs(rowterms(x))
            addterm!(terms, n, f, op)
        end
        outnames = keys(emptyvalue(s))
        outnames in seen && return
        outnames in visiting && throw(
            ArgumentError(
                "$op: summarizer dependency cycle through $(repr(first(outnames)))"),
        )
        push!(visiting, outnames)          # a stack: the walk is depth-first
        for d in dependencies(s)
            expand(d)
        end
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
    return Tuple(protos), Tuple(requested), termtuple(terms)
end

# The input's column element types, by name.
chunktypes(c::DataFrame) =
    NamedTuple{Tuple(propertynames(c))}(Tuple(eltype(col) for col in eachcol(c)))
chunktypes(nt::NamedTuple) = map(eltype, nt)

# The summarizers' input for one chunk: its column table plus a column per row
# term, each computed behind `termvalues`' function barrier. Only the
# summarizers read it, so term columns never reach a transform's output.
function terminput(c::DataFrame, terms::NamedTuple, op::String)
    nt = Tables.columntable(c)
    isempty(terms) && return nt
    for n in keys(terms)
        haskey(nt, n) && throw(
            ArgumentError("$op row term $(repr(n)) collides with an input column"),
        )
    end
    return merge(nt, map(f -> termvalues(f, nt), terms))
end

# A term's column, typed by inference over the row type rather than by the
# values, so it is the same for every chunk and never abstract when inference
# succeeds (`r.x > 2 ? r.x : 1` is `Union{Float64,Int64}`, not `Real`). When
# inference gives up (a closure over an untyped global, say), the values type
# it, as `addcolumns`' `rowvalues` does. The term sees each row as the
# NamedTuple its type was inferred over, the row a count window evaluates its
# own terms on, so a term types and behaves alike hoisted or not.
function termvalues(f::F, nt::NamedTuple) where {F}
    R = termrowtype(typeof(nt))
    T = termtype(f, R)
    # Comprehensions, not `collect(T, generator)`: on 1.10 the generator's
    # closure holds `R` as a bare `DataType` and every row dispatches. When
    # 1.10 support is dropped, recheck whether one generator suffices.
    is = eachindex(first(nt))
    return concreteterm(T) ? T[f(rowat(R, nt, i)) for i in is] :
           [f(rowat(R, nt, i)) for i in is]
end

# A term's type over rows of type `R`, the one inference both `termvalues` and
# an unhoisted count window (over `storerowtype(intypes)`) use.
termtype(f, ::Type{R}) where {R<:NamedTuple} = Base.promote_op(f, R)

# The row type of a column table, generated so it stays a constant past the
# 32 columns where `storerowtype(chunktypes(nt))` stops inferring. The `eltype`
# calls are emitted, not made by the generator, which would see only the
# methods defined before it (a column type's own `eltype` could be newer).
@generated function termrowtype(::Type{NamedTuple{N,C}}) where {N,C}
    eltypes = [:(eltype($(fieldtype(C, j)))) for j in 1:fieldcount(C)]
    return :(NamedTuple{N,Tuple{$(eltypes...)}})
end

# A type a term may store unboxed: concrete, or a Union of concrete types.
# Recursing over the Union, not `Base.uniontypes`, lets it fold to a constant.
concreteterm(T::Union) = concreteterm(T.a) && concreteterm(T.b)
concreteterm(T) = isconcretetype(T)

# The columns kept from the summarizers' input by the transforms that store
# its rows (`addrollingcolumns`, `summarizewindows`): `:time`, the keys and
# every column a summarizer reads, or `nothing` (keep all) when one may read
# any. A stored row carries every kept column, so an unread column costs a copy
# per row, and an unread `Union{Missing,T}` one boxes every row (issue #98).
storedcolumns(protos::Tuple, keycols) = readcolumns(protos, Symbol[:time; keycols])

# `cols` and the union of `protos`' `inputcolumns`, or `nothing` when one may
# read any column.
function readcolumns(protos::Tuple, cols::Vector{Symbol} = Symbol[])
    for s in protos
        ins = inputcolumns(s)
        ins === nothing && return nothing
        union!(cols, ins)
    end
    return Tuple(cols)
end

# The input projected to the stored columns, in the input's order. A missing
# column is left for `fresh` to report, as it would be without the projection.
projectinput(nt::NamedTuple, ::Nothing) = nt
projectinput(nt::NamedTuple, cols::Tuple) =
    NamedTuple{Tuple(k for k in keys(nt) if k in cols)}(nt)

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
# type from the promoted schema, the state type from the prototypes), so the
# kernels specialize on them. The cycle-closing transforms (`summarizecycles`,
# `intervalize`) close tables through both buffers; `summarizewindows`' refold
# tier reuses the pool.
mutable struct GroupTable{K,S<:Tuple}
    table::KeyDict{K,S}
    scratch::Vector{Pair{K,S}}  # key-ordered emission buffer, reused
    pool::Vector{S}             # retired state tuples, zeroed on reuse
end

GroupTable{K,S}() where {K,S<:Tuple} =
    GroupTable{K,S}(KeyDict{K,S}(), Pair{K,S}[], S[])

# A summarizing fold's groups, built for the first schema by the transform's
# grouping: the state tuple itself when keyless (`Val(())`), or a `GroupTable`
# keyed by the key columns' promoted types, which every row's key converts to.
# A declared `KeySet` makes `DenseGroups` (below).
newgroups(::Val{()}, stateprotos::Tuple, ::NamedTuple) = map(fresh, stateprotos)
newgroups(keynames::Val, stateprotos::S, types::NamedTuple) where {S<:Tuple} =
    GroupTable{storekeytype(types, keynames),S}()

# The groups rebuilt for a widened schema. `stateprotos` must already be
# widened: it fixes a new table's value type, which an empty table couldn't. A
# table's keys widen with the schema too (a key column may come to admit
# `missing`), and its pool is dropped, since retired tuples have the old type.
widengroups(states::Tuple, ::Tuple, intypes::NamedTuple, ::Val{()}) =
    widenstates(states, intypes)
function widengroups(gt::GroupTable, stateprotos::Tuple, intypes::NamedTuple,
    keynames::Val)
    widened = newgroups(keynames, stateprotos, intypes)
    for (k, gs) in gt.table
        widened.table[k] = widenstates(gs, intypes)
    end
    return widened
end

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

@inline summaryrow(t, states::Tuple, outs::Val) =
    merge((; time = t), summaryvalues(states, outs))
@inline summaryrow(t, k::NamedTuple, states::Tuple, outs::Val) =
    merge((; time = t), k, summaryvalues(states, outs))

# The key names ride in a Val so the key's NamedTuple type is known statically.
# The key is typed by the row's declared field types, as the group table's is,
# not by its values' runtime types, which are narrower once a key column admits
# `missing` (issue #114). Base's selection keeps a NamedTuple row's field types;
# a `Tables.rows` row's are its columns' eltypes.
@inline keyvalues(row::NamedTuple, ::Val{KN}) where {KN} = NamedTuple{KN}(row)
@generated function keyvalues(row::Tables.ColumnsRow{T}, ::Val{KN}) where {T,KN}
    K = NamedTuple{KN,Tuple{(eltype(fieldtype(T, c)) for c in KN)...}}
    vals = map(c -> :(Tables.getcolumn(row, $(QuoteNode(c)))), KN)
    return Expr(:block, Expr(:meta, :inline), Expr(:new, K, vals...))
end

# Groups are emitted in key order: the key's values compared lexicographically,
# as tuples are. Generated field by field rather than through `values`, whose
# tuple of a key with a `missing`-admitting field isn't inferred, so a sort
# would dispatch every comparison.
@generated function keyisless(a::NamedTuple{KN}, b::NamedTuple{KN}) where {KN}
    field(t, c) = :(getfield($t, $(QuoteNode(c))))
    isempty(KN) && return false
    ex = :(isless($(field(:a, last(KN))), $(field(:b, last(KN)))))
    for c in reverse(KN[1:(end-1)])
        x, y = field(:a, c), field(:b, c)
        ex = :(isless($x, $y) || (isequal($x, $y) && $ex))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end
@inline groupisless(a::Pair, b::Pair) = keyisless(first(a), first(b))

# For the once-per-run drains (`summarize`'s and `lastrow`'s flush); the
# per-cycle drain reuses `closecycle!`'s buffer. `lastrow` passes a bare KeyDict.
sortedgroups(d::AbstractDict) = sort!(collect(d); lt = groupisless)
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
promotedvaluetype(::Type{S}, protos::Tuple, outs::Val) where {S} =
    promotefields(valuetype(S, outs), typeof(emptyvalues(protos, outs)))

# Field-wise promotion of the value type VT with the NamedTuple type E of the
# same names, or their Union should VT not be a concrete NamedTuple.
function promotefields(::Type{VT}, ::Type{E}) where {VT,E<:NamedTuple}
    (VT <: NamedTuple && isconcretetype(VT) && fieldnames(VT) == fieldnames(E)) ||
        return Union{VT,E}
    return NamedTuple{fieldnames(E),
        Tuple{
            ntuple(i -> promote_type(fieldtype(VT, i), fieldtype(E, i)),
                fieldcount(E))...,
        }}
end

# The value type of a count window (`Bars`, `barwindow`), which emits all
# `missing` until it holds its count: VT promoted with an all-missing row, by
# the same path as the empty values above.
missingfields(::Type{VT}, ::Val{R}) where {VT,R} =
    promotefields(VT, NamedTuple{R,NTuple{length(R),Missing}})

# The all-missing row of a count window's value type V.
missingrow(::Type{V}, ::Val{R}) where {V,R} =
    convert(V, NamedTuple{R}(ntuple(_ -> missing, Val(length(R)))))

# A keyed grid row (`:time`, the key, then the values) and its type, for paths
# that emit empty values beside summaries.
gridrow(t, k, v) = merge((; time = t), k, v)
gridrowtype(::Type{T}, ::Type{K}, ::Type{V}) where {T,K,V} =
    Base.promote_op(gridrow, T, K, V)

# A grid's row type, over value type V, and its empty row.
gridtypes(::Type{T}, ::Type{K}, ::Type{V}, protos::Tuple,
    outs::Val) where {T,K,V} =
    gridrowtype(T, K, V), convert(V, emptyvalues(protos, outs))

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

newgroups(ks::KeySet, stateprotos::Tuple, ::NamedTuple) =
    densegroups(stateprotos, length(ks.keys))
widengroups(dg::DenseGroups, ::S, intypes::NamedTuple, ::KeySet) where {S<:Tuple} =
    DenseGroups{S}(S[widenstates(gs, intypes) for gs in dg.states], dg.folded)

# A declared key set's key names, which a row's key is looked up by.
declarednames(::KeySet{NamedTuple{KN,T}}) where {KN,T} = Val(KN)

# The dense row type and empty row. Type-unstable setup, once per chunk.
densetypes(::Type{T}, stateprotos::S, protos::Tuple, ::KeySet{K},
    outs::Val) where {T,S,K} =
    gridtypes(T, K, promotedvaluetype(S, protos, outs), protos, outs)

@inline function densefold!(dg::DenseGroups, ks::KeySet, row)
    i = keyindex(ks, keyvalues(row, declarednames(ks)))
    updateall!(@inbounds(dg.states[i]), row)
    @inbounds dg.folded[i] = true
    return nothing
end

# Emit one row per declared key at `t` (the summary for a slot that folded
# rows, else the empty values), zeroing the folded slots once their values are
# copied out. Allocates only the rows it pushes.
function closedense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet, t, outs::Val,
    emptyrow) where {RT}
    states, folded = dg.states, dg.folded
    for i in eachindex(ks.keys, states, folded)
        k = @inbounds ks.keys[i]
        if @inbounds folded[i]
            gs = @inbounds states[i]
            push!(rows, convert(RT, gridrow(t, k, summaryvalues(gs, outs))))
            @inbounds states[i] = freshall!(gs)
            @inbounds folded[i] = false
        else
            push!(rows, convert(RT, gridrow(t, k, emptyrow)))
        end
    end
    return rows
end

# A summarizing transform's configuration, held in `SummaryFold`'s abstract
# field, so the chunk iterators and per-chunk drivers compile once for every
# summarizer set.
struct FoldConfig{P<:Tuple,G,O}
    protos::P
    grouping::G  # summaryspec's: the key names' `Val`, or a KeySet
    outs::Val{O}
end

# Per-run state shared by the summarization transforms (and intervalize), in
# fields because reassigned closure captures are boxed. The state types depend
# on the schema, so these fields are untyped; per-row work sits behind the
# kernels' function barrier, which dispatch on the groups' type.
mutable struct SummaryFold
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    stateprotos::Any       # state tuple serving as the template for fresh copies
    groups::Any            # a state tuple, GroupTable or DenseGroups (`newgroups`)
    cycletime::Any         # summarizecycles: the open cycle's time
    const cfg::FoldConfig  # abstract, a dispatch per chunk
    const keycols::Vector{Symbol}  # checked against the first chunk
    const terms::NamedTuple  # the row terms, from `prototypes`
    const newcols::Tuple{Vararg{Symbol}}  # output columns appended to the input
    const op::String       # the transform, for messages
end
SummaryFold(cfg::FoldConfig, keycols::Vector{Symbol}, terms::NamedTuple,
    op::String; newcols::Tuple{Vararg{Symbol}} = ()) =
    SummaryFold(nothing, nothing, nothing, nothing, cfg, keycols, terms, newcols,
        op)

# Per-chunk setup: check the key columns, then that the output columns
# `newcols` are new, against the first chunk, promote the schema, then build
# the states and groups on the first chunk, by `grouping` (`summaryspec`'s), or
# widen them when the promotion changes. Returns the column table, with the row
# terms' columns.
function preparechunk!(fold::SummaryFold, c::DataFrame)
    (; protos, grouping) = fold.cfg
    if fold.types === nothing
        checkkeycolumns(fold.keycols, c, fold.op)
        checknewcolumns(c, fold.newcols, fold.op)
    end
    nt = terminput(c, fold.terms, fold.op)
    types = promotetypes(fold.types, chunktypes(nt))
    if fold.types === nothing
        fold.stateprotos = newstates(protos, types)
        fold.groups = newgroups(grouping, fold.stateprotos, types)
    elseif types != fold.types
        fold.stateprotos = widenstates(fold.stateprotos, types)
        fold.groups = widengroups(fold.groups, fold.stateprotos, types, grouping)
    end
    fold.types = types
    return nt
end

# --- folding kernels -------------------------------------------------------
#
# Called once per chunk with concretely typed arguments, so each specializes on
# the groups, state tuple and column table. The kernels are written once over
# the kinds of groups the transforms build (`newgroups`): a keyless state tuple,
# a `GroupTable` or `DenseGroups`, with the transform's grouping beside them,
# reaching them through helpers with a method per kind (`rowstates!`,
# `foldrow!`, `closerows`, `closegroups!`).

function foldall!(states::Tuple, nt::NamedTuple)
    for row in Tables.rows(nt)
        updateall!(states, row)
    end
    return nothing
end

# A row's states: the keyless state tuple itself, or its key's in the table.
@inline rowstates!(states::Tuple, ::Tuple, ::Val{()}, row) = states
@inline rowstates!(gt::GroupTable{K,S}, stateprotos::S, keynames::Val,
    row) where {K,S} = groupstates!(gt, keyvalues(row, keynames), stateprotos)

# Fold a row into its group: its states, or its declared key's slot.
@inline foldrow!(groups, stateprotos::Tuple, grouping, row) =
    updateall!(rowstates!(groups, stateprotos, grouping, row), row)
@inline foldrow!(dg::DenseGroups, ::Tuple, ks::KeySet, row) =
    densefold!(dg, ks, row)

# The closing kernels' output rows of times `T`, and the empty row a dense
# close emits for a key without rows (`nothing` for a sparse table, whose
# close emits present keys only).
@inline closerows(::GroupTable{K,S}, ::Type{T}, ::S, protos::Tuple, ::Val,
    outs::Val) where {K,S,T} = (rowtype(T, K, S, outs)[], nothing)
@inline function closerows(::DenseGroups, ::Type{T}, stateprotos::Tuple,
    protos::Tuple, ks::KeySet, outs::Val) where {T}
    RT, emptyrow = densetypes(T, stateprotos, protos, ks, outs)
    return RT[], emptyrow
end

# Close every group at `t`: the present keys of a table (`closecycle!`), or
# every declared key (`closedense!`).
@inline closegroups!(rows, gt::GroupTable, ::Val, t, outs::Val, ::Nothing) =
    closecycle!(rows, gt, t, outs)
@inline closegroups!(rows, dg::DenseGroups, ks::KeySet, t, outs::Val, emptyrow) =
    closedense!(rows, dg, ks, t, outs, emptyrow)

# `summarize`'s fold: every row into its group's states.
function foldgroups!(groups, stateprotos::Tuple, nt::NamedTuple, grouping)
    for row in Tables.rows(nt)
        foldrow!(groups, stateprotos, grouping, row)
    end
    return nothing
end

# `summarizecycles`' fold. A cycle closes when a later time arrives, so the open
# cycle carries across chunks and the last is closed by `flushcycle`. Returns
# the closed cycles' rows, the groups and the open cycle's time; keyless states
# are zeroed by `freshall!`, which may replace them. Keyed, a cycle emits its
# present keys; over declared keys, one row per declared key.
function foldcycles!(states::S, ::Tuple, nt::NamedTuple, cycletime, ::Val{()},
    protos::Tuple, outs::Val) where {S<:Tuple}
    rows = rowtype(eltype(nt.time), S, outs)[]
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            # summaryrow copied the values out, so the states are zeroed and
            # reused.
            cycletime === nothing ||
                push!(rows, summaryrow(something(cycletime), states, outs))
            cycletime = t
            states = freshall!(states)
        end
        updateall!(states, row)
    end
    return rows, states, cycletime
end
function foldcycles!(groups, stateprotos::Tuple, nt::NamedTuple, cycletime,
    grouping, protos::Tuple, outs::Val)
    rows, emptyrow =
        closerows(groups, eltype(nt.time), stateprotos, protos, grouping, outs)
    for row in Tables.rows(nt)
        t = row.time
        if cycletime === nothing || t != cycletime
            cycletime === nothing || closegroups!(rows, groups, grouping,
                something(cycletime), outs, emptyrow)
            cycletime = t
        end
        foldrow!(groups, stateprotos, grouping, row)
    end
    return rows, groups, cycletime
end

# Close the open cycle, at time `t`, once the stream ends.
flushcycle(states::S, ::Tuple, t, ::Val{()}, protos::Tuple,
    outs::Val) where {S<:Tuple} = [summaryrow(t, states, outs)]
function flushcycle(groups, stateprotos::Tuple, t, grouping, protos::Tuple,
    outs::Val)
    rows, emptyrow =
        closerows(groups, typeof(t), stateprotos, protos, grouping, outs)
    return closegroups!(rows, groups, grouping, t, outs, emptyrow)
end

# The present keys sorted into a reused buffer.
function sortinto!(scratch::Vector{<:Pair}, groups::AbstractDict)
    empty!(scratch)
    append!(scratch, groups)
    return sort!(scratch; lt = groupisless)
end

# Retire every state tuple to the pool (`groupstates!` zeroes them on reuse)
# and clear the table.
function retireall!(gt::GroupTable)
    for states in values(gt.table)
        push!(gt.pool, states)
    end
    empty!(gt.table)
    return nothing
end

# Emit one row per present key, in key order, then retire the table. Allocates
# only the rows it pushes.
function closecycle!(rows, gt::GroupTable, t, outs::Val)
    for (k, states) in sortinto!(gt.scratch, gt.table)
        push!(rows, summaryrow(t, k, states, outs))
    end
    retireall!(gt)
    return rows
end

# `addsummarycolumns`' fold: each row's group's values once it is folded.
function foldrunning!(groups, stateprotos::S, nt::NamedTuple, n::Int, grouping,
    outs::Val) where {S<:Tuple}
    vals = Vector{valuetype(S, outs)}(undef, n)
    i = 0
    for row in Tables.rows(nt)
        states = rowstates!(groups, stateprotos, grouping, row)
        updateall!(states, row)
        vals[i+=1] = summaryvalues(states, outs)
    end
    return vals
end

# --- summarization transforms ----------------------------------------------

# The setup every summarizing transform shares: the validated key columns and
# declared `keyset` (a `KeySet`, or `nothing`), the expanded summarizers with
# their requested output names and row terms, and the names as the kernels'
# `Val`s. A `grid` transform's keyless output is dense over `nokeyset`. The
# `grouping` is what the fold's groups are built by (`newgroups`): the declared
# key set if any, else the key names. `cfg` bundles what a `SummaryFold` reads.
function summaryspec(summarizers, key, op::String; keyset = nothing,
    grid::Bool = false)
    keycols = keycolumns(key, op)
    protos, requested, terms = prototypes(tosummarizers(summarizers), keycols, op)
    keyed = !isempty(keycols)
    keynames = Val(Tuple(keycols))
    ks = tokeyset(keyset, keycols, op)   # a keyless `keyset` is an error
    grid && !keyed && (ks = nokeyset(op))
    outs = Val(requested)
    return (; keycols, protos, requested, terms, ks, keynames, outs,
        cfg = FoldConfig(protos, something(ks, keynames), outs))
end

# A chunk of output rows, or nothing when there are none.
rowsframe(rows::AbstractVector) = isempty(rows) ? nothing : rowsdataframe(rows)

# Rows as a DataFrame of fresh columns. Concretely typed rows are unzipped by
# generated code and handed to DataFrames' vector-of-columns constructor, which
# compiles once, where `DataFrame(rows)` compiles Tables.jl's row-to-column
# path per row type.
rowsdataframe(rows::AbstractVector) = DataFrame(rows)
@generated function rowsdataframe(rows::AbstractVector{NamedTuple{N,T}}) where {N,T}
    (isempty(N) || !isconcretetype(NamedTuple{N,T})) && return :(DataFrame(rows))
    names = Expr(:ref, :Symbol, map(QuoteNode, N)...)
    return :(DataFrame(unzipcolumns(rows), $names; copycols = false))
end

@generated function unzipcolumns(rows::AbstractVector{NamedTuple{N,T}}) where {N,T}
    cols = [Symbol(:col, j) for j in eachindex(N)]
    allocs = [:($(cols[j]) = Vector{$(fieldtype(T, j))}(undef, n)) for j in eachindex(N)]
    sets = [:(@inbounds $(cols[j])[i] = getfield(r, $j)) for j in eachindex(N)]
    return quote
        n = length(rows)
        $(allocs...)
        for (i, r) in enumerate(rows)
            $(sets...)
        end
        return AbstractVector[$(cols...)]
    end
end

# The grid of empty values for no data: a row per time per declared key, the
# empty key for a keyless grid.
function emptygrid(times, keys, e::NamedTuple)
    RT = gridrowtype(eltype(times), eltype(keys), typeof(e))
    return rowsframe(RT[convert(RT, gridrow(t, k, e)) for t in times for k in keys])
end

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
    (; keycols, terms, cfg) = summaryspec(summarizers, key, "summarize")
    # Folds the whole run and emits at its stop.
    return pipetransform(; startfree = false, stopfree = false) do upstream, ctx
        fold = SummaryFold(cfg, keycols, terms, "summarize")
        return chunkmap(c -> summarizechunk!(fold, c), upstream;
            flush = () -> summarizeflush(fold, ctx.stop))
    end
end
summarize(p::CausalPipeline, summarizers; kwargs...) =
    summarize(summarizers; kwargs...)(p)

function summarizechunk!(fold::SummaryFold, c::DataFrame)
    nt = preparechunk!(fold, c)
    foldgroups!(fold.groups, fold.stateprotos, nt, fold.cfg.grouping)
    return nothing
end

function summarizeflush(fold::SummaryFold, stop)
    cfg = fold.cfg
    groups = fold.groups
    groups isa GroupTable && return sortedrows(groups, stop, cfg.outs)
    isempty(fold.keycols) || return nothing   # keyed, and no data
    vals =
        groups === nothing ? emptyvalues(cfg.protos, cfg.outs) :
        summaryvalues(groups, cfg.outs)
    return rowsdataframe([merge((; time = stop), vals)])
end

# `summarize`'s keyed output, a row per key, sorted.
sortedrows(gt::GroupTable, stop, outs::Val) =
    rowsframe([summaryrow(stop, k, gs, outs) for (k, gs) in sortedgroups(gt)])

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
    (; keycols, terms, cfg) =
        summaryspec(summarizers, key, "summarizecycles"; keyset)
    return pipetransform() do upstream, ctx
        fold = SummaryFold(cfg, keycols, terms, "summarizecycles")
        return chunkmap(c -> cyclechunk!(fold, c), upstream;
            flush = () -> cycleflush(fold))
    end
end
summarizecycles(p::CausalPipeline, summarizers; kwargs...) =
    summarizecycles(summarizers; kwargs...)(p)

function cyclechunk!(fold::SummaryFold, c::DataFrame)
    (; protos, grouping, outs) = fold.cfg
    nt = preparechunk!(fold, c)
    rows, fold.groups, fold.cycletime = foldcycles!(fold.groups,
        fold.stateprotos, nt, fold.cycletime, grouping, protos, outs)
    return rowsframe(rows)
end

function cycleflush(fold::SummaryFold)
    fold.cycletime === nothing && return nothing
    (; protos, grouping, outs) = fold.cfg
    return rowsframe(
        flushcycle(fold.groups, fold.stateprotos, fold.cycletime, grouping,
            protos, outs),
    )
end

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
    (; keycols, requested, terms, cfg) =
        summaryspec(summarizers, key, "addsummarycolumns")
    # Cumulative from the run's start.
    return pipetransform(; startfree = false) do upstream, ctx
        # hidden dependencies are never added, so only the requested outputs
        # are checked as new columns
        fold = SummaryFold(cfg, keycols, terms, "addsummarycolumns";
            newcols = requested)
        return chunkmap(c -> runningchunk!(fold, c), upstream)
    end
end
addsummarycolumns(p::CausalPipeline, summarizers; kwargs...) =
    addsummarycolumns(summarizers; kwargs...)(p)

function runningchunk!(fold::SummaryFold, c::DataFrame)
    nt = preparechunk!(fold, c)
    vals = foldrunning!(fold.groups, fold.stateprotos, nt, nrow(c),
        fold.cfg.grouping, fold.cfg.outs)
    return hcat(c, rowsdataframe(vals); copycols = false)
end
