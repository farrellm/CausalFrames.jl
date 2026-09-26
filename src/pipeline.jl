"""
    CausalPipeline(run)

A lazy description of how to produce time-series data. `run` maps a
[`Context`](@ref) to a single-pass iterator of DataFrame chunks whose `:time`
is non-decreasing within and across chunks. Nothing runs until the pipeline is
evaluated by [`load`](@ref), [`stream`](@ref) or [`scan`](@ref).

Build pipelines from sources (such as [`clock`](@ref) or [`readcsv`](@ref))
and chain transforms with `|>`:

```jldoctest
ticks = DataFrame(time = [1, 2, 3], bid = [10.0, -1.0, 10.4], ask = [10.2, 10.3, 10.6])
p = readtable(ticks) |>
    filterrows(r -> r.bid > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
load(Context(0, 10), p)

# output

CausalFrame{Int64} with 2 rows over [0, 10]
 Row │ time   bid      ask      mid
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     10.0     10.2     10.1
   2 │     3     10.4     10.6     10.5
```

Every operator is *causal*: its output at time `t` depends only on input rows
with time `<= t`.
"""
struct CausalPipeline{F}
    run::F
end

"""
    load(ctx::Context, p::CausalPipeline) -> CausalFrame
    load(ctx::Context) -> (CausalPipeline -> CausalFrame)

Evaluate `p` over `ctx` and materialize the result as a [`CausalFrame`](@ref).
The only evaluation that holds the whole window in memory; the frame wraps the
produced chunks without copying them. An empty result gives a zero-row frame
with only `:time`.

The curried form ends a chain: `p |> load(ctx)` is `load(ctx, p)`.
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

Evaluate `p` over `ctx` for its side effects, discarding each chunk as it is
produced — the way to run a pipeline ending in a writer such as
[`writecsv`](@ref). Chunks are validated as by [`load`](@ref).

The curried form ends a chain: `p |> scan(ctx)` is `scan(ctx, p)`.
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
# order and window bounds. Within-chunk order and schema equality are left to
# the chunk protocol (sources validate their input, transforms preserve order),
# so the public constructor's O(n) scans are skipped. Returns the chunk's last
# time, to be passed back as `prev`.
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

Evaluate `p` over `ctx` incrementally, yielding one [`CausalFrame`](@ref) per
chunk. Chunk sizes are set by the source (for example `clock`'s `batchsize` or
`readcsv`'s `chunkbytes`). Transforms carry their state across chunks, so
concatenating the streamed frames equals `load(ctx, p)`.

The frames' contexts tile the window: frame `i` covers `[bᵢ₋₁, bᵢ)`, where
`b₀ = ctx.start` and `bᵢ` is the first time of chunk `i + 1`; the last frame
extends to `ctx.stop`. The iterator is single-pass and looks one chunk ahead.

The curried form ends a chain: `p |> stream(ctx)` is `stream(ctx, p)`.
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
    # The O(1) guards of `checkchunk`: window bounds here, cross-chunk order at
    # the lookahead below. substart is the chunk's first time, except for the
    # first chunk, where it is ctx.start.
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

# Skips the public constructor's O(n) validation, as `checkchunk` explains.
trustedframe(ctx::Context{T}, chunk::DataFrame) where {T} =
    CausalFrame{T}(Trusted(), ctx, [chunk])
