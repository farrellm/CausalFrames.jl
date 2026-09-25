# The window tiers shared by addrollingcolumns and summarizewindows. The
# expanded summarizer set is partitioned per accumulator, so the weakest
# structure in a call no longer sets the cost of the rest:
#
# - running: GroupSummarizers whose windowed state (`freshwindowed`) is
#   invertible. Per-key running states, update!d as rows arrive and downdate!d,
#   oldest first, as they leave: O(1) amortized per row.
# - tree: the other MonoidSummarizers, plus any group whose realized state
#   defeats `isinvertible`. Per-key segment trees (segtree.jl): O(log window).
# - refold: everything else. Fresh states folded over the window's buffered
#   rows: O(window).
# - derived: states with no fields (the dependent summarizers). They fold
#   nothing, so they need no per-key copy and no tier; one instance serves
#   every key and window, reading its dependencies' values at emission.
#
# At emission the tiers' state tuples are spliced back into the topological
# order by a compile-time permutation (`mergestates`), and the ordinary
# `summaryvalues` runs over the result, so a dependent reads dependencies from
# any tier. Every tier holds the same rows per key, so any one of them decides
# whether a window is empty. Partitioning is type-unstable setup, run once per
# run and on each widening; everything per row sees concretely typed tier
# tuples, with an absent tier's structure `nothing` and its templates `()`, so
# a single-tier call compiles to the single-algorithm kernel it used to be.

const RUNNINGTIER = 1
const TREETIER = 2
const REFOLDTIER = 3
const DERIVEDTIER = 4
const TIERNAMES = (:running, :tree, :refold, :derived)

# The partition of one realized state set: each tier's template tuple and, per
# topological position, the (tier, index) pair it moved to. The running
# templates are windowed states; the others are ordinary ones.
struct Tiering{SR<:Tuple,ST<:Tuple,SF<:Tuple,SD<:Tuple,P}
    running::SR
    tree::ST
    refold::SF
    derived::SD
    perm::Val{P}
end

# A state with no fields holds nothing to fold. The one exception is a set made
# only of such states (a custom summarizer that ignores its rows), which then
# keeps its structural tier so that some tier still tracks which windows are
# empty.
function tiering(protos::Tuple, intypes::NamedTuple)
    states = newstates(protos, intypes)
    windowed = map(s -> freshwindowed(s, intypes), protos)
    stateless = map(st -> Base.issingletontype(typeof(st)), states)
    folding = !all(stateless)
    codes = map(protos, windowed, stateless) do p, w, sl
        folding && sl ? DERIVEDTIER :
        p isa GroupSummarizer && isinvertible(w) ? RUNNINGTIER :
        p isa MonoidSummarizer ? TREETIER : REFOLDTIER
    end
    counts = zeros(Int, length(TIERNAMES))
    perm = map(c -> (c, counts[c] += 1), codes)
    pick(c, src) = Tuple(src[i] for i in eachindex(codes) if codes[i] == c)
    return Tiering(pick(RUNNINGTIER, windowed), pick(TREETIER, states),
        pick(REFOLDTIER, states), pick(DERIVEDTIER, states), Val(perm))
end

# Each state's tier, in topological order — for tests and for reasoning about a
# call, not used on any path.
tiernames(::Tiering{SR,ST,SF,SD,P}) where {SR,ST,SF,SD,P} =
    map(c -> TIERNAMES[first(c)], P)

# The topologically ordered state tuple, from the four tiers' tuples in tier
# order. Generated so the permutation is spliced in as constant field accesses:
# the result is a tuple of references to the tiers' (mutable) states, built
# without allocating.
@generated function mergestates(::Val{P}, sources::Tuple) where {P}
    fields = [:(getfield(getfield(sources, $t), $j)) for (t, j) in P]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, fields...))
end

mergedstates(tg::Tiering) =
    mergestates(tg.perm, (tg.running, tg.tree, tg.refold, tg.derived))

# The emitted value type, from the merged templates. Windowed states report the
# same value type as ordinary ones (the `freshwindowed` contract), so this is
# the type every tier combination emits.
tieredvaluetype(tg::Tiering, protos::Tuple, outs::Val) =
    promotedvaluetype(typeof(mergedstates(tg)), protos, outs)

# Append an admitted row to the shared buffer, when the call keeps one.
@inline pushrow!(::Nothing, row) = nothing
@inline pushrow!(buffer::Vector, row) = (push!(buffer, row); nothing)

# --- running groups ---------------------------------------------------------

# One key's running window state: the states with every in-window row folded
# in, and how many rows that is. A group is deleted when its last row leaves,
# so presence in the table implies live >= 1 and an absent key means an empty
# window. Deleted groups retire to the table's pool and are zeroed on reuse —
# the `GroupTable` idiom — so a key whose window empties and refills does not
# rebuild its states (the windowed Min/First states own vectors).
mutable struct RunningGroup{S<:Tuple}
    states::S
    live::Int
end

struct RunningTable{K,S<:Tuple}
    groups::Dict{K,RunningGroup{S}}
    pool::Vector{RunningGroup{S}}
end

RunningTable{K,S}() where {K,S<:Tuple} =
    RunningTable{K,S}(Dict{K,RunningGroup{S}}(), RunningGroup{S}[])

# Fold `row` into key `k`'s group, making the group if needed. A declared
# keyset (summarizewindows) is checked only there: a key that already has a
# group was declared.
@inline function admitgroup!(rt::RunningTable{K,S}, stateprotos::S, k, row,
    ks::Union{Nothing,KeySet} = nothing, keynames::Val = Val(())) where {K,S<:Tuple}
    pool = rt.pool
    # Without a keyset the closure captures only the pool and templates — one
    # capturing the row (a String key, say) measurably slows every admission.
    g =
        ks === nothing ? get!(() -> claimgroup!(pool, stateprotos), rt.groups, k) :
        get!(rt.groups, k) do
            checkdeclared(ks, row, keynames)
            claimgroup!(pool, stateprotos)
        end
    updateall!(g.states, row)
    g.live += 1
    return nothing
end

# A zeroed group from the pool, or a new one; off the per-row path.
@noinline function claimgroup!(pool::Vector{RunningGroup{S}},
    stateprotos::S) where {S}
    isempty(pool) && return RunningGroup(map(fresh, stateprotos), 0)
    r = pop!(pool)
    r.states = freshall!(r.states)
    return r
end

# Remove `row`, the oldest row of key `k`'s group, retiring the group with it.
@inline function evictgroup!(rt::RunningTable, k, row)
    g = rt.groups[k]
    downdateall!(g.states, row)
    g.live -= 1
    if g.live == 0
        delete!(rt.groups, k)
        push!(rt.pool, g)
    end
    return nothing
end

# Fold the live rows (from head on) back into a fresh table; a function
# barrier so the per-row work is concretely typed.
function replayrunning!(rt::RunningTable{K,S}, rows::Vector, head::Int,
    stateprotos::S, keynames::Val) where {K,S<:Tuple}
    for j in head:length(rows)
        row = @inbounds rows[j]
        admitgroup!(rt, stateprotos, keyvalues(row, keynames), row)
    end
    return rt
end

# Push the live rows (from head on) into per-key trees, converting each to the
# trees' row type; a function barrier like replayrunning!.
function replaytrees!(trees::Dict{K,SegTree{S,R,T}}, rows::Vector, head::Int,
    stateprotos::S, keynames::Val) where {K,S<:Tuple,R,T}
    for j in head:length(rows)
        row = convert(R, @inbounds rows[j])
        tr = get!(() -> newsegtree(stateprotos, R, T), trees,
            keyvalues(row, keynames))
        treepush!(tr, stateprotos, row)
    end
    return trees
end

# The old trees' live rows as one time-ordered buffer, for a rebuild that needs
# a buffer where the old tiers kept none. The built-in states only ever demote,
# so this takes a custom state that turns invertible when its column widens.
# The stable sort keeps each key's rows in stream order, which is all its
# running group sees.
function treebuffer(::Type{R}, trees::AbstractDict) where {R}
    rows = R[]
    for (_, tr) in trees, j in tr.head:length(tr.rows)
        push!(rows, convert(R, @inbounds tr.rows[j]))
    end
    return sort!(rows; by = r -> r.time, alg = Base.Sort.DEFAULT_STABLE)
end

# Replay every live row of the old trees; a tree owns its rows, so this is the
# rebuild source when no buffer holds them.
function replayoldtrees!(trees::Dict{K,SegTree{S,R,T}}, old::AbstractDict,
    stateprotos::S, keynames::Val) where {K,S<:Tuple,R,T}
    for (_, tr) in old
        replaytrees!(trees, tr.rows, tr.head, stateprotos, keynames)
    end
    return trees
end
