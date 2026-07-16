# Internal streaming machinery. The chunk protocol: a pipeline's run(ctx)
# returns a single-pass lazy iterator of non-empty DataFrame chunks with
# non-decreasing times within and across chunks; consumers take ownership of
# yielded chunks. Empty chunks are filtered out here so downstream code may
# assume every chunk has at least one row.

# A source iterator driven by a stateful producer closure.
# produce() -> Union{Nothing, DataFrame}; nothing means exhausted (and must
# keep returning nothing on subsequent calls).
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

# Iteration-state sentinel: this ChunkMap's flush has already run.
struct Flushed end

"""
    chunkmap(step, upstream; flush = () -> nothing)

Lazy chunk-to-chunk transformer: applies `step(chunk::DataFrame) ->
Union{Nothing, DataFrame}` to each upstream chunk, skipping `nothing` and
empty results; once upstream is exhausted, calls `flush() ->
Union{Nothing, DataFrame}` exactly once and yields its result if non-empty.
Single-pass: `step` and `flush` may close over mutable per-run state.
Internal.
"""
chunkmap(step, upstream; flush = () -> nothing) = ChunkMap(step, flush, upstream)

struct ChunkMap{S,F,U}
    step::S
    flush::F
    upstream::U
end

Base.IteratorSize(::Type{<:ChunkMap}) = Base.SizeUnknown()
Base.eltype(::Type{<:ChunkMap}) = DataFrame

# The upstream state is wrapped in Some so it can never be confused with this
# ChunkMap's own Flushed sentinel (the upstream may itself be a ChunkMap).
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
