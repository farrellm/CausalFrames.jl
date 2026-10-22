# The window tiers shared by addrollingcolumns and summarizewindows. The
# expanded summarizer set is partitioned per accumulator, so each accumulator
# gets the cheapest algorithm its structure allows:
#
# - running: GroupSummarizers whose windowed state (`freshwindowed`) is
#   invertible. Per-key running states, update!d as rows arrive and downdate!d,
#   oldest first, as they leave: O(1) amortized per row.
# - monoid: the other MonoidSummarizers, plus any group whose realized state
#   defeats `isinvertible`. Per-key two-stack queues (monoidqueue.jl): O(1)
#   amortized per row.
# - refold: everything else. Fresh states folded over the window's buffered
#   rows: O(window).
# - derived: states with no fields (the dependent summarizers). They fold
#   nothing, so they need no per-key copy and no tier; one instance serves
#   every key and window, reading its dependencies' values at emission.
#
# At emission the tiers' state tuples are spliced back into topological order
# by a compile-time permutation (`mergestates`) and the ordinary
# `summaryvalues` runs over the result, so a dependent reads dependencies from
# any tier. Every tier holds the same rows per key, so any one decides whether
# a window is empty. Partitioning is type-unstable setup, run per run and per
# widening; per-row code sees concretely typed tier tuples, an absent tier
# being `nothing` with templates `()`, so a single-tier call compiles to a
# single-algorithm kernel.

const RUNNINGTIER = 1
const MONOIDTIER = 2
const REFOLDTIER = 3
const DERIVEDTIER = 4
const TIERNAMES = (:running, :monoid, :refold, :derived)

# The partition of one realized state set: each tier's template tuple and, per
# topological position, its (tier, index). The running templates are windowed
# states; the others are ordinary ones.
struct Tiering{SR<:Tuple,ST<:Tuple,SF<:Tuple,SD<:Tuple,P}
    running::SR
    monoid::ST
    refold::SF
    derived::SD
    perm::Val{P}
end

# A state with no fields holds nothing to fold and takes the derived tier,
# unless every state is fieldless (a custom summarizer ignoring its rows): then
# they keep their structural tiers, so some tier tracks which windows are empty.
function tiering(protos::Tuple, intypes::NamedTuple)
    states = newstates(protos, intypes)
    windowed = map(s -> freshwindowed(s, intypes), protos)
    stateless = map(st -> Base.issingletontype(typeof(st)), states)
    folding = !all(stateless)
    codes = map(protos, windowed, stateless) do p, w, sl
        folding && sl ? DERIVEDTIER :
        p isa GroupSummarizer && isinvertible(w) ? RUNNINGTIER :
        p isa MonoidSummarizer ? MONOIDTIER : REFOLDTIER
    end
    counts = zeros(Int, length(TIERNAMES))
    perm = map(c -> (c, counts[c] += 1), codes)
    return tiered(Val(perm), windowed, states)
end

# The barrier past the runtime permutation: with it a constant, each tier's
# templates are picked by `tierpick` without inferring a growing tuple.
tiered(perm::Val, windowed::Tuple, states::Tuple) =
    Tiering(tierpick(perm, Val(RUNNINGTIER), windowed),
        tierpick(perm, Val(MONOIDTIER), states), tierpick(perm, Val(REFOLDTIER), states),
        tierpick(perm, Val(DERIVEDTIER), states), perm)

# The elements of `src` whose tier is `C`, in topological order.
@generated function tierpick(::Val{P}, ::Val{C}, src::Tuple) where {P,C}
    return Expr(:tuple, (:(getfield(src, $i)) for i in eachindex(P) if P[i][1] == C)...)
end

# Each state's tier, in topological order: `barwindow` refuses a refold tier
# by it, and the tests read it.
tiernames(::Tiering{SR,ST,SF,SD,P}) where {SR,ST,SF,SD,P} =
    map(c -> TIERNAMES[first(c)], P)

# The topologically ordered state tuple, from the four tiers' tuples. Generated
# so the permutation becomes constant field accesses, building a tuple of
# references to the tiers' states without allocating.
@generated function mergestates(::Val{P}, sources::Tuple) where {P}
    fields = [:(getfield(getfield(sources, $t), $j)) for (t, j) in P]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, fields...))
end

mergedstates(tg::Tiering) =
    mergestates(tg.perm, (tg.running, tg.monoid, tg.refold, tg.derived))

# The emitted value type, from the merged templates. Windowed states have the
# ordinary states' value type (the `freshwindowed` contract), so every tier
# combination emits it.
tieredvaluetype(tg::Tiering, protos::Tuple, outs::Val) =
    promotedvaluetype(typeof(mergedstates(tg)), protos, outs)

# A count window's value type, which is all `missing` until the window fills
# (`Bars`, `barwindow`).
barsvaluetype(tg::Tiering, outs::Val) =
    missingfields(valuetype(typeof(mergedstates(tg)), outs), outs)

# Append an admitted row to the shared buffer, when the call keeps one.
@inline pushrow!(::Nothing, row) = nothing
@inline pushrow!(buffer::Vector, row) = (push!(buffer, row); nothing)

# --- running groups ---------------------------------------------------------

# One key's running window state: the states with every in-window row folded
# in, and how many rows that is. A group is deleted when its last row leaves,
# so an absent key means an empty window. Deleted groups retire to the pool and
# are zeroed on reuse, as in `GroupTable`, so a key whose window empties and
# refills doesn't rebuild its states (windowed Min/First states own vectors).
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

# Fold `row` into key `k`'s group, making it if needed. A declared keyset
# (summarizewindows) is checked only then, since a key with a group was
# declared.
@inline function admitgroup!(rt::RunningTable{K,S}, stateprotos::S, k, row,
    ks::Union{Nothing,KeySet} = nothing, keynames::Val = Val(())) where {K,S<:Tuple}
    pool = rt.pool
    # Without a keyset the closure doesn't capture the row, which measurably
    # slows every admission (with a String key, say).
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

# Evict a row from a running table, absent (`nothing`) without a running tier;
# `evictrow!` keys it first.
@inline evictkey!(::Nothing, k, row) = nothing
@inline evictkey!(rt::RunningTable, k, row) = evictgroup!(rt, k, row)
@inline evictrow!(::Nothing, row, keynames::Val) = nothing
@inline evictrow!(rt::RunningTable, row, keynames::Val) =
    evictgroup!(rt, keyvalues(row, keynames), row)

# Window w's running table, of a per-window tuple (homogeneous, so the Int
# index is type-stable), or `nothing` without a running tier.
@inline windowtable(::Nothing, w::Int) = nothing
@inline windowtable(tables::Tuple, w::Int) = @inbounds tables[w]

# Advance a time window's eviction head past the buffered rows older than its
# look-back at t, evicting each from the window's running table. No buffer
# holds no rows.
@inline evicthead!(::Nothing, head::Int, t, lookback, running, keynames::Val) =
    head
@inline function evicthead!(buffer::Vector, head::Int, t, lookback, running,
    keynames::Val)
    while head <= length(buffer) && t - @inbounds(buffer[head]).time > lookback
        evictrow!(running, @inbounds(buffer[head]), keynames)
        head += 1
    end
    return head
end

# Fold the live rows (from head on) into a fresh table; a function barrier.
function replayrunning!(rt::RunningTable{K,S}, rows::Vector, head::Int,
    stateprotos::S, keynames::Val) where {K,S<:Tuple}
    for j in head:length(rows)
        row = @inbounds rows[j]
        admitgroup!(rt, stateprotos, keyvalues(row, keynames), row)
    end
    return rt
end

# Push the live rows (from head on) into per-key queues, converting each to
# the queues' row type; a function barrier like replayrunning!.
function replaymonoids!(monoids::Dict{K,Q}, rows::Vector, head::Int,
    stateprotos::S, keynames::Val,
    lookbacks::Tuple) where {K,S<:Tuple,R,Q<:MonoidQueues{S,R}}
    for j in head:length(rows)
        row = convert(R, @inbounds rows[j])
        mq = get!(() -> newqueues(Q, stateprotos, length(lookbacks)), monoids,
            keyvalues(row, keynames))
        queuepush!(mq, row, lookbacks)
    end
    return monoids
end

# The queues' live rows as one time-ordered buffer, for a rebuild that needs a
# buffer where the previous tiers kept none. Built-in states only demote, so
# only a custom state that becomes invertible on widening gets here. The stable
# sort keeps each key's rows in stream order.
function monoidbuffer(::Type{R}, monoids::AbstractDict) where {R}
    rows = R[]
    for (_, mq) in monoids, row in liverows(mq)
        push!(rows, convert(R, row))
    end
    return sort!(rows; by = r -> r.time, alg = Base.Sort.DEFAULT_STABLE)
end

# The shared row buffer of a tiering being built: `nothing` when neither the
# running nor the refold tier needs one (the queues own their rows); empty on a
# first build; after a widening, the previous buffer converted to the new row
# type, or gathered from the previous queues (`monoidbuffer`) if there was none.
function rebuildbuffer(tg::Tiering, ::Type{R}, old) where {R}
    isempty(tg.running) && isempty(tg.refold) && return nothing
    old === nothing && return R[]
    old.buffer === nothing && return monoidbuffer(R, old.monoids)
    return convert(Vector{R}, old.buffer)
end

# The monoid tier of a tiering being built: `nothing` without monoid states,
# else per-key queues (one per look-back) replayed from the previous queues'
# live rows if any, else from the previous buffer's live rows (from `head`),
# else (a first build) empty.
function rebuildmonoids(tg::Tiering, ::Type{K}, ::Type{R}, ::Type{T}, old,
    head::Int, keynames::Val, lookbacks::Tuple) where {K,R,T}
    isempty(tg.monoid) && return nothing
    monoids =
        Dict{K,queuestype(tg.monoid, R, T)}()
    if old !== nothing && old.monoids !== nothing
        replayoldmonoids!(monoids, old.monoids, tg.monoid, keynames, lookbacks)
    elseif old !== nothing && old.buffer !== nothing
        replaymonoids!(monoids, old.buffer, head, tg.monoid, keynames, lookbacks)
    end
    return monoids
end

# Replay every live row of the previous queues, which own their rows.
function replayoldmonoids!(monoids::Dict, old::AbstractDict, stateprotos::Tuple,
    keynames::Val, lookbacks::Tuple)
    for (_, mq) in old
        replaymonoids!(monoids, mq.rows, oldestlo(mq) - mq.dropped, stateprotos,
            keynames, lookbacks)
    end
    return monoids
end

# Drop a row buffer's `dead` prefix, the rows no window still holds, once it is
# at least half the buffer: amortized O(1) per row. Returns how many rows were
# dropped.
@inline function compactdead!(buffer::Vector, dead::Int)
    (dead >= 64 && 2 * dead >= length(buffer)) || return 0
    deleteat!(buffer, 1:dead)
    return dead
end
