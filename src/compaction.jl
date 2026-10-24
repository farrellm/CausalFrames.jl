# The dead-prefix rule shared by every FIFO buffer that pops by advancing a
# head: the window transforms' row buffer, the monoid queues' rows, the
# windowed extremes' deque, `WindowValues` and `futurejoin`'s key buffers.

# The fewest dead entries worth a compaction. It spares a buffer of a few live
# rows a compaction every few pops, which cost a steady 1-row FIFO 3-5x per
# row with none; from 32 up the cost is flat. Most of these buffers are per
# key, and each may hold up to `COMPACTMIN` dead rows, so it is no larger.
const COMPACTMIN = 32

# Drop `buffer`'s first `dead` entries when they are all of it, or when they
# number at least `COMPACTMIN` and at least half of it, so compaction is
# amortized O(1) per row. Returns how many were dropped. A prefix `deleteat!`
# only advances the vector's start, and a later push past the end reuses the
# freed front rather than reallocating; moving the live rows down with
# `copyto!` instead measured 7-14% slower on `futurejoin`, for the same
# allocations.
@inline function compactdead!(buffer::Vector, dead::Int)
    dead == length(buffer) || (dead >= COMPACTMIN && 2 * dead >= length(buffer)) ||
        return 0
    dropfront!(buffer, dead)
    return dead
end

# Drop the first `dead` entries, as `compactdead!` decided, from it or from a
# vector kept aligned with its buffer. `empty!` when that is all of them: a
# whole-range `deleteat!` still advances the vector's start, through a call
# that `empty!` doesn't make.
@inline dropfront!(v::Vector, dead::Int) =
    dead == length(v) ? empty!(v) : deleteat!(v, 1:dead)
