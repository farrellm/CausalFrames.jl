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
p = readcsv("ticks.csv") |>
    filterrows(r -> r.price > 0) |>
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

Evaluate the pipeline over the time window `ctx`, materializing the whole
window into a frame. This is the only operation that forces the full window
into memory; the frame wraps the streamed chunks as-is, without copying.
An empty result yields a zero-row frame with only a `:time` column.
"""
function load(ctx::Context, p::CausalPipeline)
    chunks = DataFrame[]
    for c in p.run(ctx)
        push!(chunks, c)
    end
    return CausalFrame(ctx, chunks)
end

"""
    stream(ctx::Context, p::CausalPipeline) -> iterator of CausalFrames

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
"""
stream(ctx::Context, p::CausalPipeline) = FrameStream(ctx, p.run(ctx))

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
    next = iterate(fs.chunks, ustate)
    next === nothing &&
        return (CausalFrame(Context{T}(substart, fs.ctx.stop), chunk), nothing)
    nextchunk, nustate = next
    b = first(nextchunk.time)
    last(chunk.time) <= b || throw(ArgumentError(
        "chunk times must be non-decreasing across chunk boundaries"))
    return (CausalFrame(Context{T}(substart, b), chunk), (nextchunk, b, nustate))
end
