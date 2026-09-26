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
