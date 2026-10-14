# Internal streaming machinery. The chunk protocol: a pipeline's run(ctx)
# returns a single-pass lazy iterator of non-empty DataFrame chunks with
# non-decreasing times within and across chunks, and consumers own the chunks
# they are given. The iterators here drop empty chunks.

# A source iterator over a stateful producer closure
# produce() -> Union{Nothing, DataFrame}. Once it returns nothing (exhausted),
# it must keep doing so.
struct ChunkSource{F}
    produce::F
end

Base.IteratorSize(::Type{<:ChunkSource}) = Base.SizeUnknown()
Base.eltype(::Type{<:ChunkSource}) = DataFrame

function Base.iterate(it::ChunkSource, _ = nothing)
    while true
        out = it.produce()
        out === nothing && return nothing
        nrow(out) > 0 && return (out, nothing)
    end
end

# Pulls an iterator one element at a time for a stateful owner, such as a
# producer draining its input or a binary transform pulling its second stream.
# A mutable struct rather than a captured local, which would be boxed when
# reassigned. Sticky: once the iterator is exhausted, `pull!` returns `nothing`
# without touching it again. `state` is untyped but touched once per chunk, so
# callers annotate what they pull (`::DataFrame`) where inference needs it.
mutable struct PullCursor{I}
    const iter::I
    state::Any
    started::Bool
    done::Bool
end
PullCursor(iter::I) where {I} = PullCursor{I}(iter, nothing, false, false)

# The next element, or `nothing` once the iterator is exhausted.
function pull!(c::PullCursor)
    c.done && return nothing
    next = c.started ? iterate(c.iter, c.state) : iterate(c.iter)
    c.started = true
    if next === nothing
        c.done = true
        return nothing
    end
    x, c.state = next
    return x
end

# Iteration-state sentinel: this ChunkMap's flush has already run.
struct Flushed end

"""
    chunkmap(step, upstream; flush = () -> nothing)

Internal lazy chunk transformer. Applies `step(chunk::DataFrame) ->
Union{Nothing, DataFrame}` to each upstream chunk, skipping `nothing` and empty
results, then calls `flush() -> Union{Nothing, DataFrame}` once after upstream
ends and yields its result if non-empty. Single-pass, so `step` and `flush` may
close over mutable per-run state.
"""
chunkmap(step, upstream; flush = () -> nothing) = ChunkMap(step, flush, upstream)

struct ChunkMap{S,F,U}
    step::S
    flush::F
    upstream::U
end

Base.IteratorSize(::Type{<:ChunkMap}) = Base.SizeUnknown()
Base.eltype(::Type{<:ChunkMap}) = DataFrame

# The upstream state is wrapped in Some so it can't be confused with this
# ChunkMap's Flushed sentinel (the upstream may itself be a ChunkMap).
Base.iterate(it::ChunkMap) = advance(it, iterate(it.upstream))
Base.iterate(it::ChunkMap, s::Some) = advance(it, iterate(it.upstream, something(s)))
Base.iterate(::ChunkMap, ::Flushed) = nothing

function advance(it::ChunkMap, next)
    while next !== nothing
        chunk, ustate = next
        out = it.step(chunk)
        out isa DataFrame && nrow(out) > 0 && return (out, Some(ustate))
        next = iterate(it.upstream, ustate)
    end
    out = it.flush()
    out isa DataFrame && nrow(out) > 0 && return (out, Flushed())
    return nothing
end

# Splits one single-pass chunk iterator into two, for a consumer reading the
# same stream twice at different paces (`addrollingcolumns` summarizing its own
# input). A side with queued chunks reads its queue; otherwise it pulls
# upstream and queues the chunk for the other side as a private index over the
# same vectors (sound by the `writecsv` hand-off argument), so each side owns
# what it is given. A queue holds only the chunks the other side is ahead by.
# `keep` is the second side's view of a chunk (the chunk, a slice of it, or
# `nothing` to skip it), applied to every chunk that side gets: the first side
# can then run ahead through rows the second never reads without queueing them.
struct ChunkTee{I,F}
    upstream::PullCursor{I}
    queues::NTuple{2,Vector{DataFrame}}
    keep::F
end

function teesides(iter, keep = identity)
    tee = ChunkTee(PullCursor(iter), (DataFrame[], DataFrame[]), keep)
    return (ChunkSource(() -> teepull!(tee, 1)),
        ChunkSource(() -> teepull!(tee, 2)))
end

function teepull!(tee::ChunkTee, side::Int)
    q = tee.queues[side]
    isempty(q) || return popfirst!(q)
    while true
        c = pull!(tee.upstream)
        c === nothing && return nothing
        c = c::DataFrame
        if side == 1
            k = tee.keep(c)
            k === nothing || push!(tee.queues[2], DataFrame(k; copycols = false))
            return c
        end
        push!(tee.queues[1], DataFrame(c; copycols = false))
        k = tee.keep(c)
        k === nothing || return k
    end
end

# The clock cursor: one boundary time at a time over clk.run(ctx), pulling
# clock chunks lazily. The type-unstable pull is confined here; reading
# boundaries from the concrete `times` is typed. Clock order is trusted to the
# chunk protocol.
mutable struct IntervalCursor{T}
    const chunks::PullCursor
    times::Vector{T}
    pos::Int
end
IntervalCursor{T}(chunks) where {T} = IntervalCursor{T}(PullCursor(chunks), T[], 1)

function nextboundary!(cur::IntervalCursor{T}) where {T}
    while cur.pos > length(cur.times)
        chunk = pull!(cur.chunks)
        chunk === nothing && return nothing
        cur.times = convert(Vector{T}, chunk.time)
        cur.pos = 1
    end
    b = @inbounds cur.times[cur.pos]
    cur.pos += 1
    return b
end

# Every boundary has been handed out and the clock is exhausted.
exhausted(cur::IntervalCursor) = cur.chunks.done && cur.pos > length(cur.times)

# Pull boundaries onto `buf` until its last one is strictly past `tmax` (so
# every boundary a row up to `tmax` needs is known) or the clock is exhausted;
# `tmax = nothing` drains the clock. Shared by intervalize's boundaries and
# summarizewindows' ticks. Type-unstable, once per chunk.
function pullpast!(buf::Vector{T}, cur::IntervalCursor{T}, tmax) where {T}
    while tmax === nothing || isempty(buf) || @inbounds(buf[end]) <= tmax
        b = nextboundary!(cur)
        b === nothing && return nothing
        push!(buf, b)
    end
    return nothing
end
