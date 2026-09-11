# The monoid segment tree behind the rolling and window tree modes: an
# implicit array-based tree whose leaves are single admitted rows and whose
# inner nodes hold the combine! of their children, so any contiguous row
# range — in particular a trailing window — folds from O(log n) partial
# combinations instead of every row. Rows only append (times are
# non-decreasing) and only expire logically from the front (`head`); expired
# leaves stay in place, excluded from queries, until a capacity-triggered
# rebuild drops them — which is also what keeps a leaf poisoned by an
# absorbing value (missing, NaN) harmless once it expires, since a query never
# touches a node unless its whole range is inside the window.
#
# Appending and recombining are separate steps. `treeappend!` writes only the
# leaf; `treesync!` recombines the ancestors of every leaf appended since the
# last sync, level by level, in O(new leaves + log cap). addrollingcolumns
# queries after every append, so it syncs one leaf at a time (`treepush!`,
# O(log cap) per row); summarizewindows queries only at clock ticks, so it
# appends a tick's rows as bare leaves and syncs them together — about one
# combine per row.

# S is the state-tuple type (one state per expanded summarizer prototype), R
# the stored row type, T the time type. Leaves j = 1..cap sit at node
# cap + j - 1; node i's children are 2i and 2i + 1; cap is a power of two.
# Only nodes over appended leaves are maintained. A leaf slot past
# length(rows) holds whatever a rebuild left there, and the append that claims
# it zeroes it first; an inner node whose range lies wholly past length(rows)
# is never read, because the sync combines the right edge with `ident` instead
# and a query stops at length(rows).
# accl/accr are the query walk's two order-preserving accumulators, owned by
# the tree and zeroed per query rather than allocated: a query runs once per
# output row per window, so allocating them there cost one heap allocation per
# state per row — the single largest per-row allocation in the package.
mutable struct SegTree{S<:Tuple,R,T}
    rows::Vector{R}    # this key's admitted rows, in time order
    times::Vector{T}   # rows[j].time, aligned; for the window-start search
    nodes::Vector{S}   # length 2cap; index 1 is the root
    accl::S            # query scratch: left-edge accumulator
    accr::S            # query scratch: right-edge accumulator
    ident::S           # fresh states, never mutated: the right-edge identity
    cap::Int
    head::Int          # 1 + expired-prefix length; only ever advances
    synced::Int        # leaves 1:synced have up-to-date ancestors
end

newsegtree(stateprotos::S, ::Type{R}, ::Type{T}) where {S<:Tuple,R,T} =
    SegTree{S,R,T}(R[], T[], [map(fresh, stateprotos) for _ in 1:8],
        map(fresh, stateprotos), map(fresh, stateprotos),
        map(fresh, stateprotos), 4, 1, 0)

# Peeled explicitly rather than through `foreach`/`map`: the three-argument
# `foreach` folds over a `zip` and does not unroll for tuples, which costs a
# runtime dispatch per node on this hot path, and `map` would build a
# throwaway NTuple{n,Nothing}.
@inline combinenodes!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function combinenodes!(dest::S, a::S, b::S) where {S<:Tuple}
    combine!(first(dest), first(a), first(b))
    return combinenodes!(Base.tail(dest), Base.tail(a), Base.tail(b))
end

# Append one row as a bare leaf: zero the slot it claims and fold the row in,
# leaving the ancestors to the next `treesync!`. A full tree rebuilds first,
# which also drops the expired prefix; between rebuilds at least half the
# capacity is appended, so the rebuild amortizes to O(1) per row.
function treeappend!(tr::SegTree{S,R}, stateprotos::S,
    row::R) where {S<:Tuple,R}
    length(tr.rows) == tr.cap && rebuild!(tr, stateprotos)
    push!(tr.rows, row)
    push!(tr.times, row.time)
    i = tr.cap + length(tr.rows) - 1
    leaf = freshall!(@inbounds tr.nodes[i])
    @inbounds tr.nodes[i] = leaf
    updateall!(leaf, row)
    return nothing
end

# Recombine the ancestors of the leaves appended since the last sync, one level
# at a time over the contiguous run of parents they share: O(new leaves +
# log cap), no allocation. The run starts at the head rather than at the first
# unsynced leaf when the head is past it: a node reaching into the expired
# prefix is never queried, so neither is a stale value it carries. Only the
# run's last parent can have a right child wholly past length(rows), and that
# child is unmaintained, so the parent combines with `ident` instead.
function treesync!(tr::SegTree)
    n = length(tr.rows)
    lo = max(tr.synced + 1, tr.head)
    if lo <= n
        nodes = tr.nodes
        l = tr.cap + lo - 1
        r = tr.cap + n - 1
        while l > 1
            last = r          # the last maintained node on the child level
            l >>>= 1
            r >>>= 1
            for i in l:(r-1)
                @inbounds combinenodes!(nodes[i], nodes[2i], nodes[2i+1])
            end
            @inbounds combinenodes!(nodes[r], nodes[2r],
                2r + 1 <= last ? nodes[2r+1] : tr.ident)
        end
    end
    tr.synced = n
    return nothing
end

# Append one row and recombine its ancestors at once — O(log cap), for a caller
# that queries after every append.
function treepush!(tr::SegTree{S,R}, stateprotos::S, row::R) where {S<:Tuple,R}
    treeappend!(tr, stateprotos, row)
    treesync!(tr)
    return nothing
end

# Fold rows lo:hi (1-based, inclusive; caller has checked lo <= hi and synced
# the tree) into a state tuple: the standard bottom-up walk, kept
# order-preserving with two accumulators — accL collects left-edge nodes left
# to right, accR right-edge nodes right to left — because First/Last combine
# correctly only over stream-ordered ranges.
#
# The accumulators are the tree's own scratch, zeroed here rather than
# allocated, so the returned tuple is **borrowed**: it stays valid only until
# the next query on this tree. Every caller (`emittree!`, `windowstates`) reads
# it straight through `summaryvalues`, which copies the values out.
function treequery(tr::SegTree{S}, lo::Int, hi::Int) where {S<:Tuple}
    accl = freshall!(tr.accl)
    accr = freshall!(tr.accr)
    tr.accl = accl
    tr.accr = accr
    l = tr.cap + lo - 1
    r = tr.cap + hi          # one past the last leaf: the walk is half-open
    while l < r
        if isodd(l)
            @inbounds combinenodes!(accl, accl, tr.nodes[l])
            l += 1
        end
        if isodd(r)
            r -= 1
            @inbounds combinenodes!(accr, tr.nodes[r], accr)
        end
        l >>>= 1
        r >>>= 1
    end
    combinenodes!(accl, accl, accr)
    return accl
end

# The first live index whose row is inside a window ending at t: the least
# m in head:length(times) with t - times[m] <= lb, or length + 1 when the
# window is empty. The predicate is the rolling kernel's own membership test
# verbatim — never rearranged to times[m] >= t - lb, which could disagree at
# the last ulp for floating-point times — and it is monotone in m because
# times are non-decreasing, so the search is a plain binary chop.
function windowstart(times::Vector{T}, head::Int, t, lb) where {T}
    lo, hi = head, length(times) + 1
    while lo < hi
        m = (lo + hi) >>> 1
        if t - @inbounds(times[m]) <= lb
            hi = m
        else
            lo = m + 1
        end
    end
    return lo
end

# Drop the expired prefix and re-lay the live rows at a capacity leaving at
# least live + 1 free slots, so rebuilds stay amortized-O(1) per append even
# when nothing has expired.
#
# At an unchanged capacity the live leaves' state tuples move to the front by
# reference, and nothing is folded, zeroed or recombined here: the expired
# tuples they swap with land past the new length(rows), where each is zeroed by
# the append that claims its slot, and `synced = 0` has the next sync recombine
# the ancestors. That is the steady state under a window short enough to
# expire rows as fast as they arrive: `live` stays small, `cap` settles, and a
# rebuild fires every few appends. Rebuilding by allocation there once cost
# O(cap) fresh states each time — about six allocations per admitted row on a
# 25-unit window — and re-zeroing and recombining the whole node vector still
# cost several state operations per append. Only a capacity change allocates,
# re-folding the live rows into fresh leaves.
function rebuild!(tr::SegTree{S}, stateprotos::S) where {S<:Tuple}
    dead = tr.head - 1
    deleteat!(tr.rows, 1:dead)
    deleteat!(tr.times, 1:dead)
    live = length(tr.rows)
    cap = max(4, nextpow(2, 2 * (live + 1)))
    if cap == tr.cap
        nodes = tr.nodes
        for i in cap:(cap+live-1)
            @inbounds nodes[i], nodes[i+dead] = nodes[i+dead], nodes[i]
        end
    else
        nodes = [map(fresh, stateprotos) for _ in 1:(2*cap)]
        for (j, row) in enumerate(tr.rows)
            updateall!(@inbounds(nodes[cap+j-1]), row)
        end
        tr.nodes = nodes
        tr.cap = cap
    end
    tr.head = 1
    tr.synced = 0
    return nothing
end
