# The monoid tier's per-key structure, behind addrollingcolumns and
# summarizewindows: a key's rows plus, per window, a Two-Stacks Lite queue of
# combine!d partial state tuples, O(1) amortized per row and per query.
#
# Two-Stacks Lite keeps one queue whose front part holds suffix combines and
# whose back part holds raw values, plus one aggregate of the back. Here the raw
# values are the key's stored rows, shared by its windows, so the partials never
# duplicate them. Per window:
#
# - `front[j]` folds rows fstart + j - 1 through fhi (sequence numbers), so the
#   window's oldest live row `lo` reads one suffix, `front[lo - fstart + 1]`;
# - `back` folds the rows after fhi;
# - the window is `front[lo …] ⊕ back`, order-preserving, since the front's
#   rows all precede the back's (First/Last combine only over stream-ordered
#   ranges).
#
# Expiry only advances `lo`. Once it passes fhi the front is used up, and the
# next query flips the live rows lo:n into a new front (one update! and one
# combine! per row) and zeroes the back; rows that expired while in the back
# are never folded again. Each row is flipped at most once per window, so a
# window costs one update! into the back plus one update! and one combine! at
# its flip per row, and a query one combine!. The front vector grows only when
# a flip needs more slots and its state tuples are reused, zeroed, by later
# flips, so a steady window allocates nothing. A query never reads a partial
# covering a row before `lo`, which is what makes an expired absorbing row
# (missing, NaN) harmless.
#
# `barwindow` has its own two-stack queue (`BarQueue`), over its own generated
# folds: it runs inside an outer state's update!/value, where these tuple folds
# would trip inference's recursion limit.

# One window's queue over the states S. Row positions are the key's sequence
# numbers (1 for its first stored row), which compaction never shifts.
mutable struct PartialQueue{S<:Tuple}
    const front::Vector{S}
    fstart::Int  # the sequence number front[1] starts at
    fhi::Int     # the last row the front covers; the back folds those after
    lo::Int      # the window's oldest live row
    back::S
    out::S       # the query's combine, borrowed until the next query
end

PartialQueue(stateprotos::S) where {S<:Tuple} =
    PartialQueue{S}(S[], 1, 0, 1, map(fresh, stateprotos), map(fresh, stateprotos))

# A key's rows, in stream order, and its windows' queues, one per look-back.
# The rows before every window's `lo` are dead; `dropped` of them have been
# deleted, so row sequence number r sits at rows[r - dropped]. The queues are a
# vector, not a tuple, so the type (and every method over it) doesn't depend on
# the window count.
mutable struct MonoidQueues{S<:Tuple,R,T}
    const rows::Vector{R}
    const times::Vector{T}  # rows[j].time, aligned; for expiry
    dropped::Int
    const queues::Vector{PartialQueue{S}}
end

# A key's empty structure for `nwindows` look-backs, its type taken from the
# store's value type so the kernels' `get!` infers it.
newqueues(::Type{MonoidQueues{S,R,T}}, stateprotos::S,
    nwindows::Int) where {S,R,T} =
    MonoidQueues{S,R,T}(R[], T[], 0,
        [PartialQueue(stateprotos) for _ in 1:nwindows])

# The store's value type.
queuestype(::S, ::Type{R}, ::Type{T}) where {S<:Tuple,R,T} = MonoidQueues{S,R,T}

# The number of rows the key has stored: the newest row's sequence number.
@inline nstored(mq::MonoidQueues) = mq.dropped + length(mq.rows)

# The oldest row any window still holds; the rows before it are dead.
@inline oldestlo(mq::MonoidQueues) = minimum(q -> q.lo, mq.queues)

# The live rows, oldest first: from the earliest window's `lo`.
liverows(mq::MonoidQueues) =
    view(mq.rows, (oldestlo(mq)-mq.dropped):length(mq.rows))

# Peeled explicitly: the three-argument `foreach` goes through a `zip` and
# doesn't unroll for tuples (a runtime dispatch per state), and `map` would
# build a throwaway NTuple{n,Nothing}.
@inline combinenodes!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function combinenodes!(dest::S, a::S, b::S) where {S<:Tuple}
    combine!(first(dest), first(a), first(b))
    return combinenodes!(Base.tail(dest), Base.tail(a), Base.tail(b))
end

# Append a row: fold it into every window's back, then expire each window
# against the row's own time. Every later query is at or after it, and a row
# outside a window at one time is outside it at every later time, so this only
# moves expiry earlier, keeping a key that is never queried from holding its
# whole history. The dead prefix is dropped once it dominates the rows.
function queuepush!(mq::MonoidQueues{S,R}, row::R,
    lookbacks::Tuple) where {S,R}
    push!(mq.rows, row)
    push!(mq.times, row.time)
    for q in mq.queues
        updateall!(q.back, row)
    end
    queueexpire!(mq, row.time, 1, lookbacks...)
    compactqueues!(mq)
    return nothing
end

# Expire every window against t, peeling the (possibly heterogeneous)
# look-backs with the window index in tow.
@inline queueexpire!(mq::MonoidQueues, t, w::Int) = nothing
@inline function queueexpire!(mq::MonoidQueues, t, w::Int, lb, rest...)
    expirequeue!(mq, w, t, lb)
    return queueexpire!(mq, t, w + 1, rest...)
end

# Expire window w against t.
@inline expirequeue!(mq::MonoidQueues, w::Int, t, lb) =
    expirewindow!(mq, @inbounds(mq.queues[w]), t, lb)

# Advance the window's `lo` past the rows outside it at t, by the kernels'
# membership test verbatim (rearranging it to time >= t - lb could disagree in
# the last ulp for float times). rolling.jl adds the Bars method, over
# `keepnewest!`.
@inline function expirewindow!(mq::MonoidQueues, q::PartialQueue, t, lb)
    lo = q.lo
    n = nstored(mq)
    times = mq.times
    d = mq.dropped
    while lo <= n && !(t - @inbounds(times[lo-d]) <= lb)
        lo += 1
    end
    q.lo = lo
    return nothing
end

# Expire the window to the key's newest n rows.
@inline function keepnewest!(mq::MonoidQueues, q::PartialQueue, n::Int)
    q.lo = max(q.lo, nstored(mq) - n + 1)
    return nothing
end

# Drop the rows no window still holds by `compactdead!`'s rule, which needs
# `COMPACTMIN` rows, tested first so a key holding fewer skips the scan over
# the windows: amortized O(1) per row.
@inline function compactqueues!(mq::MonoidQueues)
    length(mq.rows) >= COMPACTMIN || return nothing
    dead = compactdead!(mq.rows, oldestlo(mq) - mq.dropped - 1)
    dead == 0 && return nothing
    dropfront!(mq.times, dead)
    mq.dropped += dead
    return nothing
end

# Whether window w, as expired so far, holds rows.
@inline hasrows(mq::MonoidQueues, w::Int) =
    @inbounds(mq.queues[w]).lo <= nstored(mq)

# Window w's states, as expired so far: `nothing` for an empty window, else
# the window's fold, **borrowed** (the queue's own front or scratch): valid
# only until the next query on this window. Callers read it straight through
# `summaryvalues`.
@inline queuestates(mq::MonoidQueues, w::Int) =
    hasrows(mq, w) ? livestates(mq, w) : nothing

# The fold of a window known to hold rows, so that a caller that knows it
# (summarizewindows, whose emptied keys are dropped) gets a concrete tuple,
# not a Union with Nothing that would make emission dispatch.
@inline livestates(mq::MonoidQueues, w::Int) =
    livestates(mq, @inbounds mq.queues[w])
function livestates(mq::MonoidQueues, q::PartialQueue)
    n = nstored(mq)
    q.lo > q.fhi && flipqueue!(q, mq.rows, mq.dropped, n)
    f = @inbounds q.front[q.lo-q.fstart+1]
    q.fhi == n && return f
    combinenodes!(q.out, f, q.back)
    return q.out
end

# Fold the live rows lo:n into suffix combines, newest first, and zero the back.
@noinline function flipqueue!(q::PartialQueue{S}, rows::Vector, dropped::Int,
    n::Int) where {S}
    lo = q.lo
    m = n - lo + 1
    front = q.front
    while length(front) < m
        push!(front, map(fresh, q.back))
    end
    for j in m:-1:1
        st = freshall!(@inbounds front[j])
        @inbounds front[j] = st
        updateall!(st, @inbounds rows[lo+j-1-dropped])
        j < m && @inbounds combinenodes!(st, st, front[j+1])
    end
    q.back = freshall!(q.back)
    q.fstart = lo
    q.fhi = n
    return nothing
end
