"""
    CausalPipeline(run)

A lazy description of how to produce time-series data: conceptually a
function `Context -> single-pass lazy iterator of DataFrame chunks` (time
non-decreasing within and across chunks). Nothing runs until the iterator is
consumed; [`load`](@ref) drains it into a [`CausalFrame`](@ref) and
[`stream`](@ref) yields one frame per chunk. Build pipelines from sources
([`emptyframe`](@ref), [`clock`](@ref), [`readcsv`](@ref)) and chain
transforms with `|>`:

```julia
p = readcsv("ticks.csv";
        types = Dict(:time => Int, :bid => Float64, :ask => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
```

Every operator must be *causal*: its output at time `t` may depend only on
input rows with time `<= t`.
"""
struct CausalPipeline{F}
    run::F
end

"""
    load(ctx::Context, p::CausalPipeline) -> CausalFrame
    load(ctx::Context) -> (CausalPipeline -> CausalFrame)

Evaluate the pipeline over the time window `ctx`, materializing the whole
window into a frame. This is the only operation that forces the full window
into memory; the frame wraps the streamed chunks as-is, without copying.
An empty result yields a zero-row frame with only a `:time` column.

The curried form composes with `|>`, so a chain can end in its own
evaluation: `p |> load(ctx)` is equivalent to `load(ctx, p)`.
"""
function load(ctx::Context{T}, p::CausalPipeline) where {T}
    chunks = DataFrame[]
    prev = nothing
    for c in p.run(ctx)
        prev = checkchunk(ctx, c, prev)
        push!(chunks, c)
    end
    return CausalFrame{T}(Trusted(), ctx, chunks)
end
load(ctx::Context) = (p::CausalPipeline) -> load(ctx, p)

"""
    scan(ctx::Context, p::CausalPipeline) -> Nothing
    scan(ctx::Context) -> (CausalPipeline -> Nothing)

Evaluate the pipeline over `ctx`, discarding every chunk as it is produced.
Nothing is materialized — this is how to run a pipeline for its side effects
(see [`writecsv`](@ref)) without paying for a frame that would be thrown
away. Chunks are validated exactly as [`load`](@ref) validates them.

The curried form composes with `|>`, so a chain can end in its own
evaluation: `p |> scan(ctx)` is equivalent to `scan(ctx, p)`.
"""
function scan(ctx::Context, p::CausalPipeline)
    prev = nothing
    for c in p.run(ctx)
        prev = checkchunk(ctx, c, prev)
    end
    return nothing
end
scan(ctx::Context) = (p::CausalPipeline) -> scan(ctx, p)

# O(1)-per-chunk guards against a misbehaving hand-rolled source: cross-chunk
# order and window bounds. Within-chunk order and schema equality are the
# chunk protocol's responsibility (sources validate their own input;
# transforms preserve order), so the public constructor's O(n) scans are
# skipped. Returns the chunk's last time, to be passed back as `prev`.
function checkchunk(ctx::Context, c::DataFrame, prev)
    prev === nothing || prev <= first(c.time) ||
        throw(
            ArgumentError(
                "chunk times must be non-decreasing across chunk boundaries"),
        )
    first(c.time) >= ctx.start && last(c.time) <= ctx.stop ||
        throw(ArgumentError(
            "chunk times must lie in [$(ctx.start), $(ctx.stop)]"))
    return last(c.time)
end

"""
    stream(ctx::Context, p::CausalPipeline) -> iterator of CausalFrames
    stream(ctx::Context) -> (CausalPipeline -> iterator of CausalFrames)

Evaluate the pipeline over `ctx` incrementally, yielding one
[`CausalFrame`](@ref) per chunk without ever materializing the whole window.
Chunk boundaries are chosen by the pipeline's source (see e.g. the
`batchsize`/`chunkbytes` arguments of [`clock`](@ref) and
[`readcsv`](@ref)); stateful operators carry their state across chunks, so
concatenating the streamed frames equals `load` of the whole window.

The frames' contexts tile `[ctx.start, ctx.stop)`: frame `i` covers
`[bᵢ₋₁, bᵢ)` where `b₀ = ctx.start`, `bᵢ` is the first time of chunk
`i + 1`, and the last frame's context stops at `ctx.stop`. The iterator is
single-pass and maintains one chunk of lookahead.

The curried form composes with `|>`, so a chain can end in its own
evaluation: `p |> stream(ctx)` is equivalent to `stream(ctx, p)`.
"""
stream(ctx::Context, p::CausalPipeline) = FrameStream(ctx, p.run(ctx))
stream(ctx::Context) = (p::CausalPipeline) -> stream(ctx, p)

struct FrameStream{T,U}
    ctx::Context{T}
    chunks::U
end

Base.IteratorSize(::Type{<:FrameStream}) = Base.SizeUnknown()
Base.eltype(::Type{FrameStream{T,U}}) where {T,U} = CausalFrame{T}

function Base.iterate(fs::FrameStream)
    next = iterate(fs.chunks)
    next === nothing && return nothing
    chunk, ustate = next
    return emitframe(fs, chunk, fs.ctx.start, ustate)
end
Base.iterate(::FrameStream, ::Nothing) = nothing
Base.iterate(fs::FrameStream, st::Tuple) = emitframe(fs, st...)

# Emit `chunk` as a frame over [substart, b) where b is the first time of the
# next chunk (checking cross-chunk order), or over [substart, ctx.stop] when
# `chunk` is the last one.
function emitframe(fs::FrameStream{T}, chunk, substart, ustate) where {T}
    # The same O(1) guards as load: window bounds here, cross-chunk order at
    # the lookahead below (substart is the chunk's own first time except for
    # the very first chunk, where it is ctx.start).
    first(chunk.time) >= substart || throw(ArgumentError(
        "chunk times must lie in [$(fs.ctx.start), $(fs.ctx.stop)]"))
    next = iterate(fs.chunks, ustate)
    if next === nothing
        last(chunk.time) <= fs.ctx.stop || throw(
            ArgumentError(
                "chunk times must lie in [$(fs.ctx.start), $(fs.ctx.stop)]"),
        )
        return (trustedframe(Context{T}(substart, fs.ctx.stop), chunk), nothing)
    end
    nextchunk, nustate = next
    b = first(nextchunk.time)
    last(chunk.time) <= b || throw(ArgumentError(
        "chunk times must be non-decreasing across chunk boundaries"))
    return (trustedframe(Context{T}(substart, b), chunk), (nextchunk, b, nustate))
end

# Within-chunk order and schema equality are trusted to the chunk protocol
# (the O(n) part of the public constructor's validation); the cheap boundary
# and bounds guards above still run per chunk.
trustedframe(ctx::Context{T}, chunk::DataFrame) where {T} =
    CausalFrame{T}(Trusted(), ctx, [chunk])
