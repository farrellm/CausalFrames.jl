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

    CausalPipeline(run; startfree = false, stopfree = false) -> CausalPipeline

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

A pipeline read by several operators of one evaluation runs once for all of
them when that is exact; see `sharing`. The constructor builds a source
of your own. It is shared only between readers asking for the same context,
unless it declares that its rows don't depend on where the context starts or
stops.

#### Arguments
- `run`: a function `ctx -> iterator` of `DataFrame` chunks clipped to the
  context.

#### Keywords
- `startfree = false`: whether running over an earlier start and dropping the
  rows before `start` gives the rows a run from `start` gives, as for a file.
- `stopfree = false`: likewise for a later stop and the rows at or after
  `stop`. A source emitting rows at `stop` isn't `stopfree`.

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

---

    sharing(force::Bool) -> (CausalPipeline -> CausalPipeline)
    sharing(p::CausalPipeline, force::Bool) -> CausalPipeline

A transform passing its input through unchanged, but deciding whether the
operators reading the result share one run of it.

By default a pipeline read in several places of one evaluation, such as both
sides of `asofjoin(p, p |> f)` or the input of `addrollingcolumns`,
runs once for all of them when that gives each reader the rows it would get
alone: when they ask for the same context, or when the pipeline's rows don't
depend on where the context starts (or stops) and only that differs. Files and
tables don't, and most transforms keep their input's independence. A
`clock`, `head`, a cumulative `addsummarycolumns`, or a
`summarize` emitting at `stop` do depend on it, so a reader asking for
a widened context gets a run of its own.

#### Arguments
- `force`: `true` shares one run among every reader, over the union of their
  contexts, each reader seeing that run clipped to its own context: rows that
  may differ from its own run's when the pipeline depends on the context.
  `false` gives every reader its own run of the pipeline and of everything
  upstream of it, even over the same context, which holds no chunks back for
  a reader that lags far behind the others.

```julia
p = clock(1) |> addsummarycolumns(Count())
t = addrollingcolumns((w2 = 2,), Sum(:count))
DataFrame(load(Context(3, 6), p |> sharing(true) |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      3             6
   2 │     4      4             9
   3 │     5      5            12
```

Here the counts depend on where the run starts. Sharing the run widened by the
look-back shows the counts from 1, the ones summed; without `sharing(true)` the
rows come from their own run over `[3, 6)` and restart at 1.
