# Frames and pipelines

Contents: Core types, Evaluation.

## Core types

    Context(start, stop)

The time window a `CausalPipeline` is evaluated over.

#### Arguments
- `start`, `stop`: the window bounds, promoted to a common time type `T`. Any
  ordered type works (`DateTime`, `Int`, `Float64`, …); `start <= stop` is
  required.

Sources clip their output to the half-open interval `[start, stop)`. A loaded
`CausalFrame` may also hold rows at `stop`, since some transforms (such
as `summarize`) emit there.

---

    timetype(ctx::Context{T}) -> T

The time type of a context.

---

    CausalFrame(ctx::Context, chunks::Vector{DataFrame})
    CausalFrame(ctx::Context, df::DataFrame)

A materialized time-series table over the window `ctx`, usually obtained from
`load` rather than constructed directly.

#### Arguments
- `ctx`: the window the data covers.
- `chunks` / `df`: the data, as time-ordered DataFrames. The frame takes
  ownership of them; do not mutate them afterwards.

The constructor checks, throwing an `ArgumentError` otherwise, that every
non-empty chunk has a `:time` column with element type `<: T`, that all
non-empty chunks share their column names (element types may differ;
`DataFrame(frame)` promotes them), that time is non-decreasing within and
across chunks, and that every time lies in `[ctx.start, ctx.stop]`.

The backing chunks are not part of the API. Read the data through the Tables.jl
interface or `DataFrame(frame)`, and query it with `context`,
`nrow(frame)` and `names(frame)`.

---

    context(frame::CausalFrame) -> Context

The window the frame was loaded over.

---

    DataFrame(frame::CausalFrame) -> DataFrame

Copy the frame into a plain `DataFrame`, promoting column element types across
chunks. A frame with no rows gives a zero-row DataFrame with only `:time`.

## Evaluation

    CausalPipeline(run)

A lazy description of how to produce time-series data. `run` maps a
`Context` to a single-pass iterator of DataFrame chunks whose `:time`
is non-decreasing within and across chunks. Nothing runs until the pipeline is
evaluated by `load`, `stream` or `scan`.

Build pipelines from sources (such as `clock` or `readcsv`)
and chain transforms with `|>`:

```julia
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

---

    load(ctx::Context, p::CausalPipeline) -> CausalFrame
    load(ctx::Context) -> (CausalPipeline -> CausalFrame)

Evaluate `p` over `ctx` and materialize the result as a `CausalFrame`.
The only evaluation that holds the whole window in memory; the frame wraps the
produced chunks without copying them. An empty result gives a zero-row frame
with only `:time`.

The curried form ends a chain: `p |> load(ctx)` is `load(ctx, p)`.

---

    stream(ctx::Context, p::CausalPipeline) -> iterator of CausalFrames
    stream(ctx::Context) -> (CausalPipeline -> iterator of CausalFrames)

Evaluate `p` over `ctx` incrementally, yielding one `CausalFrame` per
chunk. Chunk sizes are set by the source (for example `clock`'s `batchsize` or
`readcsv`'s `chunkbytes`). Transforms carry their state across chunks, so
concatenating the streamed frames equals `load(ctx, p)`.

The frames' contexts tile the window: frame `i` covers `[bᵢ₋₁, bᵢ)`, where
`b₀ = ctx.start` and `bᵢ` is the first time of chunk `i + 1`; the last frame
extends to `ctx.stop`. The iterator is single-pass and looks one chunk ahead.

The curried form ends a chain: `p |> stream(ctx)` is `stream(ctx, p)`.

---

    scan(ctx::Context, p::CausalPipeline) -> Nothing
    scan(ctx::Context) -> (CausalPipeline -> Nothing)

Evaluate `p` over `ctx` for its side effects, discarding each chunk as it is
produced — the way to run a pipeline ending in a writer such as
`writecsv`. Chunks are validated as by `load`.

The curried form ends a chain: `p |> scan(ctx)` is `scan(ctx, p)`.
