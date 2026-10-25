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
    chunkmap(step, upstream; flush = () -> nothing, abandon = nothing)

Internal lazy chunk transformer. Applies `step(chunk::DataFrame) ->
Union{Nothing, DataFrame}` to each upstream chunk, skipping `nothing` and empty
results, then calls `flush() -> Union{Nothing, DataFrame}` once after upstream
ends and yields its result if non-empty. Single-pass, so `step` and `flush` may
close over mutable per-run state. Abandoning it (`abandon!`) abandons
`upstream`, then calls `abandon()` if given, for the other inputs `step`
reads.
"""
chunkmap(step, upstream; flush = () -> nothing, abandon = nothing) =
    ChunkMap(step, flush, upstream, abandon)

struct ChunkMap{S,F,U,A}
    step::S
    flush::F
    upstream::U
    abandon::A
end

function abandon!(it::ChunkMap)
    abandon!(it.upstream)
    it.abandon === nothing || it.abandon()
    return nothing
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

# The clock cursor: one boundary time at a time over the clock's chunks, pulling
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

# A chunk's column names, checked against `prev`, the first chunk's
# (`nothing` before it), for an operator whose chunks must share one schema.
# O(ncols) per chunk. `label` names the operator in the error.
function samecolumns(prev::Union{Nothing,Vector{String}}, c::DataFrame,
    label::String)
    cols = names(c)
    prev === nothing || prev == cols ||
        throw(
            ArgumentError(
                "$label: chunk columns changed mid-stream, from $prev to $cols"),
        )
    return cols
end
