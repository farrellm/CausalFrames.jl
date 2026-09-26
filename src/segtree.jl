# The monoid segment tree behind the rolling and window tree tiers: an implicit
# array tree whose leaves are rows and whose inner nodes hold the combine! of
# their children, so a contiguous row range (a trailing window) folds from
# O(log n) partial states. Rows only append and expire logically from the front
# (`head`). Expired leaves stay in place until a rebuild drops them; a query
# only reads nodes wholly inside the window, so an expired leaf holding an
# absorbing value (missing, NaN) is harmless.
#
# `treeappend!` writes only the leaf; `treesync!` recombines the ancestors of
# every leaf appended since the last sync, in O(new leaves + log cap).
# addrollingcolumns queries after every append, so it syncs per row
# (`treepush!`, O(log cap)); summarizewindows queries only at ticks, so it syncs
# a tick's rows together, about one combine per row.

# S is the state-tuple type (one state per expanded summarizer prototype), R
# the stored row type, T the time type. Leaves j = 1..cap sit at node
# cap + j - 1; node i's children are 2i and 2i + 1; cap is a power of two.
# Only nodes over appended leaves are maintained. A leaf slot past length(rows)
# holds whatever a rebuild left there and is zeroed by the append that claims
# it; an inner node wholly past length(rows) is never read, since the sync
# combines the right edge with `ident` and queries stop at length(rows).
# accl/accr are the query's two order-preserving accumulators, owned by the tree
# and zeroed per query so queries don't allocate.
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

# Peeled explicitly: the three-argument `foreach` goes through a `zip` and
# doesn't unroll for tuples (a runtime dispatch per node), and `map` would
# build a throwaway NTuple{n,Nothing}.
@inline combinenodes!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function combinenodes!(dest::S, a::S, b::S) where {S<:Tuple}
    combine!(first(dest), first(a), first(b))
    return combinenodes!(Base.tail(dest), Base.tail(a), Base.tail(b))
end

# Append one row as a bare leaf (zeroing its slot), leaving the ancestors to
# the next `treesync!`. A full tree rebuilds first, dropping the expired
# prefix; rebuilds amortize to O(1) per row.
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

# Recombine the ancestors of the leaves appended since the last sync, level by
# level over their contiguous run of parents: O(new leaves + log cap), no
# allocation. The run starts no earlier than the head, since nodes reaching
# into the expired prefix are never queried. Only the run's last parent can
# have a right child wholly past length(rows); it combines with `ident`.
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

# Append one row and recombine its ancestors: O(log cap), for a caller that
# queries after every append.
function treepush!(tr::SegTree{S,R}, stateprotos::S, row::R) where {S<:Tuple,R}
    treeappend!(tr, stateprotos, row)
    treesync!(tr)
    return nothing
end

# Fold rows lo:hi (inclusive; the caller has checked lo <= hi and synced the
# tree) into a state tuple by the bottom-up walk, order-preserving with two
# accumulators (accl collects left-edge nodes left to right, accr right-edge
# nodes right to left), since order-sensitive states such as First/Last
# combine only over stream-ordered ranges.
#
# The returned tuple is the tree's own scratch, **borrowed**: valid only until
# the next query on this tree. Callers (`treestates` in rolling.jl,
# `tierstates` in windows.jl) read it straight through `summaryvalues`.
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

# The first live index whose row is inside a window ending at t: the least m in
# head:length(times) with t - times[m] <= lb, or length + 1 if none. The
# predicate is the rolling kernel's membership test verbatim (rearranging it to
# times[m] >= t - lb could disagree in the last ulp for float times), and it
# is monotone in m, so this is a binary search.
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
# least live + 1 free slots, so rebuilds stay amortized O(1) per append.
#
# At an unchanged capacity, the steady state under a short window, the live
# leaves' state tuples swap to the front by reference and nothing is folded,
# zeroed or recombined: the expired tuples land past length(rows), to be zeroed
# by the appends that claim them, and `synced = 0` has the next sync recombine
# the ancestors. Only a capacity change allocates, re-folding the live rows
# into fresh leaves.
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
