# The monoid segment tree behind addrollingcolumns's tree path: an implicit
# array-based tree whose leaves are single admitted rows and whose inner
# nodes hold the combine! of their children, so any contiguous row range —
# in particular a trailing window — folds from O(log n) partial combinations
# instead of every row. Rows only append (times are non-decreasing) and only
# expire logically from the front (`head`); expired leaves stay in place,
# excluded from queries, until a capacity-triggered rebuild drops them —
# which is also what keeps a leaf poisoned by an absorbing value (missing,
# NaN) harmless once it expires, since a query never touches a node unless
# its whole range is inside the window.

# S is the state-tuple type (one state per expanded summarizer prototype), R
# the stored row type, T the time type. Leaves j = 1..cap sit at node
# cap + j - 1; node i's children are 2i and 2i + 1; cap is a power of two.
# Slots past length(rows) hold fresh (identity) states, so parents are
# correct without special-casing, and the next append mutates the identity
# already in place.
mutable struct SegTree{S<:Tuple,R,T}
    rows::Vector{R}    # this key's admitted rows, in time order
    times::Vector{T}   # rows[j].time, aligned; for the window-start search
    nodes::Vector{S}   # length 2cap; index 1 is the root
    cap::Int
    head::Int          # 1 + expired-prefix length; only ever advances
end

newsegtree(stateprotos::S, ::Type{R}, ::Type{T}) where {S<:Tuple,R,T} =
    SegTree{S,R,T}(R[], T[], [map(fresh, stateprotos) for _ in 1:8], 4, 1)

@inline combinenodes!(dest::S, a::S, b::S) where {S<:Tuple} =
    (foreach(combine!, dest, a, b); nothing)

# Append one row: fold it into the identity states already waiting in its
# leaf slot, then recombine the ancestors up to the root — O(log cap), no
# allocation. A full tree rebuilds first, which also drops the expired
# prefix; between rebuilds at least half the capacity is appended, so the
# O(cap) rebuild amortizes to O(1) per row.
function treepush!(tr::SegTree{S,R}, stateprotos::S, row::R) where {S<:Tuple,R}
    length(tr.rows) == tr.cap && rebuild!(tr, stateprotos)
    push!(tr.rows, row)
    push!(tr.times, row.time)
    i = tr.cap + length(tr.rows) - 1
    updateall!(@inbounds(tr.nodes[i]), row)
    i >>>= 1
    while i >= 1
        @inbounds combinenodes!(tr.nodes[i], tr.nodes[2i], tr.nodes[2i+1])
        i >>>= 1
    end
    return nothing
end

# Fold rows lo:hi (1-based, inclusive; caller has checked lo <= hi) into a
# fresh state tuple: the standard bottom-up walk, kept order-preserving with
# two accumulators — accL collects left-edge nodes left to right, accR
# right-edge nodes right to left — because First/Last combine correctly only
# over stream-ordered ranges.
function treequery(tr::SegTree{S}, stateprotos::S, lo::Int,
    hi::Int) where {S<:Tuple}
    accl = map(fresh, stateprotos)
    accr = map(fresh, stateprotos)
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

# Drop the expired prefix and rebuild at a capacity leaving at least
# live + 1 free slots, so rebuilds stay amortized-O(1) per append even when
# nothing has expired.
function rebuild!(tr::SegTree{S}, stateprotos::S) where {S<:Tuple}
    rows = tr.rows[tr.head:end]
    times = tr.times[tr.head:end]
    live = length(rows)
    cap = max(4, nextpow(2, 2 * (live + 1)))
    nodes = [map(fresh, stateprotos) for _ in 1:(2*cap)]
    for (j, row) in enumerate(rows)
        updateall!(@inbounds(nodes[cap+j-1]), row)
    end
    for i in (cap-1):-1:1
        @inbounds combinenodes!(nodes[i], nodes[2i], nodes[2i+1])
    end
    tr.rows = rows
    tr.times = times
    tr.nodes = nodes
    tr.cap = cap
    tr.head = 1
    return nothing
end
