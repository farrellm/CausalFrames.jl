# CausalFrames

[![Build Status](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://farrellm.github.io/CausalFrames.jl/dev/)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Time-series tables for Julia: DataFrames with a monotonically non-decreasing
`time` column, built lazily from composable pipelines.

```julia
using CausalFrames, Dates

p = readcsv("ticks.csv";
        types = Dict(:time => DateTime, :bid => Float64, :ask => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

frame = load(Context(DateTime(2026, 1, 1), DateTime(2026, 2, 1)), p)
```

## Concepts

- **`Context(start, stop)`** — a time window. The time type is generic:
  `DateTime`, `Int` ticks, `Float64` seconds, anything ordered.
- **`CausalPipeline`** — a lazy description of how to produce data;
  conceptually a function from a `Context` to a stream of chunks. Built from
  sources and chained with `|>`. Evaluation is streaming end to end:
  `load(ctx, pipeline)` materializes the whole window (the only operation
  that does), `stream(ctx, pipeline)` yields frames incrementally, and
  `scan(ctx, pipeline)` runs a pipeline for its side effects, discarding
  every chunk.
- **`CausalFrame`** — a materialized table. It hides its backing storage
  (one or more time-disjoint chunks, wrapped without copying) and is
  accessed via the Tables.jl interface or `DataFrame(frame)`.

## Operators

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `concatenate(ps...)` | source | run the pipelines one after another, emitting their chunks end to end; they must be passed in time order and produce identical column names |
| `merge(ps...; batchsize)` | source | run the pipelines concurrently and interleave their rows by time; the output carries the union of their columns, `missing` where a pipeline lacks one, ties broken by argument order |
| `clock(interval)` | source | one row per `interval` in `[start, stop)` |
| `readcsv(path; types, time, rename, delim)` | source | CSV read as `String` columns (`types` opts columns into concrete types); `time` picks the time column by name or a per-row function; clipped to `[start, stop)`, read incrementally |
| `writecsv(path; queue, ...)` | transform | pass-through sink: writes each chunk to `path` as it flows by, on a background task, and yields it downstream unchanged |
| `readparquet(path; time, rename, backend)` | source | parquet file (needs `using DuckDB` or `using Parquet2`); types come from the file; the context window skips row groups that cannot be in it; `time` and `rename` as for `readcsv` |
| `writeparquet(path; queue, rowgroupsize, backend, ...)` | transform | pass-through sink (needs `using Parquet2` or `using DuckDB`): writes row groups of `rowgroupsize` rows as the stream flows by; the file is valid only once the stream is exhausted |
| `readjls(path)` | source | read back a file written by `writejls`, a record at a time, clipped to `[start, stop)` |
| `writejls(path; queue)` | transform | pass-through sink through Julia's `Serialization` stdlib: one record per chunk, so columns CSV and parquet cannot encode (fitted models, `NamedTuple`s) round-trip; tied to the Julia and package versions that wrote it |
| `filterrows(pred)` | transform | keep rows where `pred(row)` |
| `addcolumns(f)` | transform | `f(row)::NamedTuple` of new column values |
| `selectcolumns(sel...)` | transform | keep the columns matching a name, `Regex`, name predicate, or collection of those (`:time` always kept) |
| `dropcolumns(sel...)` | transform | drop the columns matching the same selector forms (`:time` never dropped) |
| `reordercolumns(sel...)` | transform | move the matching columns to the front, in the selectors' order, the rest following in the input's order (`:time` always first) |
| `summarize(ss; key)` | transform | summarize the whole window into rows at time `stop` |
| `summarizecycles(ss; key)` | transform | summarize each unique timestamp independently |
| `intervalize(clock, ss; key, closelast)` | transform | summarize over the intervals `[bₖ, bₖ₊₁)` a `clock` pipeline's times define, each emitted at its end time |
| `addsummarycolumns(ss; key)` | transform | append running summary values after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | append summaries over named trailing windows, columns prefixed `{window}_` |
| `asofjoin(right; key, tolerance, ...)` | transform | append the most recent right-pipeline row at or before each row's time |
| `lag(offset)` | transform | shift every row `offset` later in time, so time `t` carries what the input had at `t - offset`; only `:time` changes and `offset` must be non-negative |
| `settime(spec)` | transform | recompute `:time` from a column name or a per-row function; rows may only move later, and the result is re-clipped to `[start, stop)` |
| `head(n)` | transform | emit the first up to `n` rows, then stop pulling the source — it genuinely stops, so a `readcsv` behind `head(10)` reads one file chunk |
| `lastrow(; key)` | transform | emit the last row, or one per key, retimed to the window's `stop` |
| `forwardfill(sel...; key, tolerance)` | transform | replace `missing` in the selected columns with that column's last non-missing value, per key, while not older than `tolerance` |
| `fillmissing(specs...)` | transform | replace `missing` with a per-column constant, given as `name => value` pairs or a `NamedTuple` |

Each transform also has an uncurried, pipeline-first form — `filterrows(p, pred)`,
`addcolumns(p, f)`, `summarize(p, ss; key)` — equivalent to the `|>` chain
(`p |> filterrows(pred)`) for when the applied form reads clearer.

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

`writecsv` streams a pipeline to disk without materializing it, and `scan`
drives the pipeline for its side effects alone:

Parquet support is optional, through two backends: either `using DuckDB` or
`using Parquet2` enables both operators. Reading prefers DuckDB (it pushes the
window into the reader) and writing prefers Parquet2 (it streams row groups
out); `backend = :duckdb` / `:parquet2` forces the choice.

`load`, `stream` and `scan` also take a curried, context-only form, so a
chain can end in its own evaluation:

```julia
readcsv("ticks.csv"; types = Dict(:time => Int, :bid => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    writecsv("clean.csv") |>
    scan(Context(0, 10^6))
```

## Summarizers

The summarization transforms take one or more summarizers — `Count()`,
`CountDistinct(:col)`, `Sum(:col)`, `SumPower(:col, n)`, `Product(:col)`,
`DotProduct(:a, :b)`, `Moment(:col, n)`, `Mean(:col)`, `Variance(:col)`,
`Std(:col)`, `Covariance(:a, :b)`, `Correlation(:a, :b)`,
`LinearRegression(predictors, response)`, `Min(:col)`, `Max(:col)`,
`First(:col)`, `Last(:col)`, or your own `Summarizer` subtype —
and an optional `key` (one or more column names) to produce a separate
summary per unique key value. Output columns are named by suffix:
`Sum(:mid)` produces `:mid_sum`, `Min(:mid)` produces `:mid_min`, and
`SumPower(:mid, 2)` produces `:mid_sumpower_2`. `LinearRegression` is the
exception, emitting a block of columns under an optional `name` prefix.

```julia
p = readcsv("ticks.csv";
        types = Dict(:time => Int, :bid => Float64, :ask => Float64)) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]; key = :symbol)
```

`Sum`, `SumPower`, `DotProduct`, and `CountDistinct` summarize no rows as `0`
and `Product` as `1`; the rest have no identity element and yield `missing`
instead. `CountDistinct(:col)` — producing `:col_countdistinct`, always an
`Int` — is also the one summarizer that does not let a `missing` poison its
output: `missing` counts as a distinct value, because unlike a sum a distinct
count stays knowable. It is the one whose state is not O(1) either, holding the
distinct values it has seen.
`Moment(:mid, n)` — the `n`-th raw moment, producing `:mid_moment_n` — is a
*dependent* summarizer, computed from `Count()` and `SumPower(:mid, n)`;
those are folded alongside it but appear in the output only if requested
themselves. `Mean`, `Variance`, `Std`, `Covariance`, `Correlation`, and
`LinearRegression` are dependent too.

`LinearRegression` fits ordinary least squares of a response on one or more
predictors, emitting a coefficient and a t statistic per term alongside `:r2`,
`:stderr` (the residual standard error), and `:n`:

```julia
p |> addrollingcolumns((h1 = Hour(1),),
    [LinearRegression([:mid, :size], :ret; name = :m1)])
# h1_m1_n, h1_m1_r2, h1_m1_stderr, h1_m1_intercept_beta,
# h1_m1_intercept_tstat, h1_m1_mid_beta, h1_m1_mid_tstat,
# h1_m1_size_beta, h1_m1_size_tstat
```

Pass `intercept = false` to drop the constant term. The `name` prefix is what
lets two regressions share a call — without it they collide on `:n`, `:r2`,
and `:stderr`. Being dependent is what makes running several cheap: every
regression is a function of the same `Count`, `Sum`, `SumPower`, and
`DotProduct` accumulators, so overlapping predictors fold each cross product
exactly once, and a `Variance` or `Correlation` requested beside them shares
those accumulators too. A window without enough rows, or with collinear
predictors, yields `NaN` rather than raising; `:n` always reports the honest
row count.

`addrollingcolumns` summarizes named trailing windows instead of the whole
window so far: `addrollingcolumns((m5 = Minute(5), h1 = Hour(1)), Mean(:mid);
key = :symbol)` appends `m5_mid_mean` and `h1_mid_mean`, each row
summarizing the rows within its look-back (`t - lookback <= time <= t`). By
default the pipeline summarizes itself; `from` names another pipeline to
summarize. The summarized pipeline runs over a context widened backward by
the longest look-back, so the first row already sees a full window; an
empty window yields the summarizer's identity or `missing` as above.

Rolling windows pick their algorithm from the summarizers' declared
structure: `GroupSummarizer`s (`Sum`, `Mean`, …) slide a running state in
O(1) per row by subtracting exiting rows, `MonoidSummarizer`s (`Min`,
`Product`, …) fold each window from a segment tree of partial combinations
in O(log window), and summarizers declaring neither re-fold each window
from scratch — see DESIGN.md for the `combine!`/`downdate!` interface a
custom structured summarizer implements. A summarizer on one of these paths
should also implement `fresh!`, which zeroes a state in place: the transforms
zero a state tuple per cycle, per interval and per window query, so building a
new one there is a heap allocation per summarizer per row.

An output column takes its element type from the input column: `Min`, `Max`,
`First`, and `Last` reproduce it verbatim, while `Sum` and `SumPower` widen it
exactly as `Base.sum` does (`Int32` sums to `Int64`, `Float32` to `Float32`).
Summarizers are typed from the input schema, so folding a large window
allocates on the order of kilobytes — see DESIGN.md for the interface a custom
`Summarizer` implements.

## Causality

Every operator above is **causal**: its output at time `t` depends only on
input rows with time `≤ t`. That gives the *chunk-concatenation property* —
loading `[a, c)` equals concatenating the results of loading `[a, b)` and
`[b, c)` — which is what makes chunked, streaming evaluation sound.

The forward-looking operators are the deliberate exceptions. They live in the
`Acausal` submodule and are never re-exported, so opting into acausality is
always explicit:

```julia
using CausalFrames.Acausal

quotes |> futurejoin(fills; key = :symbol)  # earliest fill at or after each quote
prices |> lead(Minute(5))                   # time -> time - offset
# the permissive settime, which may move rows earlier:
prices |> CausalFrames.Acausal.settime(r -> r.exchange_time)
```

The permissive `settime` is the one exception to the submodule's export rule:
it is reached only as `CausalFrames.Acausal.settime`, never unqualified, so
that `using CausalFrames.Acausal` leaves the causal `settime` usable.

`futurejoin` mirrors `asofjoin` with the match direction inverted — the
**earliest** right row whose time is not before the left row's, `missing` where
none qualifies — and `lead` mirrors `lag`. One cost worth knowing before
reaching for it: because `futurejoin` matches the earliest qualifying row, it
buffers right rows per key until a left row consumes or outruns them, and
proving that a key has no future match drains the right stream. Worst-case
memory is therefore O(right rows), against `asofjoin`'s O(keys).

The [Recipes](https://farrellm.github.io/CausalFrames.jl/dev/recipes/) page
collects the patterns that are not obvious from the operator list: ranking
within a group, joining a reference table, and using a derived stream as one.

See [DESIGN.md](DESIGN.md) for the full design.
