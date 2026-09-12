# CausalFrames

[![Build Status](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/farrellm/CausalFrames.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/farrellm/CausalFrames.jl)
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

Each operator below links to its [API reference](https://farrellm.github.io/CausalFrames.jl/dev/api/).
Every transform also has an uncurried, pipeline-first form — `filterrows(p, pred)`,
`addcolumns(p, f)`, `summarize(p, ss; key)` — equivalent to the `|>` chain
(`p |> filterrows(pred)`) for when the applied form reads clearer.

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

### Sources

A source starts a pipeline, constructing or combining streams without touching
a file.

| Operator | Semantics |
|---|---|
| [`emptyframe()`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#emptyframe) | zero rows, just a `:time` column |
| [`concatenate(ps...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#concatenate) | run the pipelines one after another, emitting their chunks end to end; they must be passed in time order and produce identical column names |
| [`merge(ps...; batchsize)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#merge) | run the pipelines concurrently and interleave their rows by time; the output carries the union of their columns, `missing` where a pipeline lacks one, ties broken by argument order |
| [`clock(interval)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#clock) | one row per `interval` in `[start, stop)` |

### File I/O

Reading starts a pipeline; writing does not end one — every writer is a
pass-through transform, so a sink can sit in the middle of a chain.

| Operator | Semantics |
|---|---|
| [`readcsv(path; types, time, rename, delim)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readcsv) | CSV read as `String` columns (`types` opts columns into concrete types); `time` picks the time column by name or a per-row function; clipped to `[start, stop)`, read incrementally |
| [`readparquet(path; time, rename, backend)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readparquet) | parquet file (needs `using DuckDB` or `using Parquet2`); types come from the file; the context window skips row groups that cannot be in it; `time` and `rename` as for `readcsv` |
| [`readjls(path)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readjls) | read back a file written by `writejls`, a record at a time, clipped to `[start, stop)` |
| [`writecsv(path; queue, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writecsv) | writes each chunk to `path` as it flows by, on a background task |
| [`writeparquet(path; queue, rowgroupsize, backend, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writeparquet) | needs `using Parquet2` or `using DuckDB`: writes row groups of `rowgroupsize` rows as the stream flows by; the file is valid only once the stream is exhausted |
| [`writejls(path; queue)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writejls) | through Julia's `Serialization` stdlib: one record per chunk, so columns CSV and parquet cannot encode (fitted models, `NamedTuple`s) round-trip; tied to the Julia and package versions that wrote it |

Parquet support is optional, through two backends: either `using DuckDB` or
`using Parquet2` enables both operators. Reading prefers DuckDB (it pushes the
window into the reader) and writing prefers Parquet2 (it streams row groups
out); `backend = :duckdb` / `:parquet2` forces the choice.

`writecsv` streams a pipeline to disk without materializing it, and `scan`
drives the pipeline for its side effects alone. `load`, `stream` and `scan`
also take a curried, context-only form, so a chain can end in its own
evaluation:

```julia
readcsv("ticks.csv"; types = Dict(:time => Int, :bid => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    writecsv("clean.csv") |>
    scan(Context(0, 10^6))
```

### Row transformations

Driven by one row at a time: testing a row, computing values from it, and
selecting or retiming rows.

| Operator | Semantics |
|---|---|
| [`filterrows(pred)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#filterrows) | keep rows where `pred(row)` |
| [`addcolumns(f)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#addcolumns) | `f(row)::NamedTuple` of new column values |
| [`head(n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#head) | emit the first up to `n` rows, then stop pulling the source — it genuinely stops, so a `readcsv` behind `head(10)` reads one file chunk |
| [`lastrow(; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#lastrow) | emit the last row, or one per key, retimed to the window's `stop` |
| [`lag(offset)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#lag) | shift every row `offset` later in time, so time `t` carries what the input had at `t - offset`; only `:time` changes and `offset` must be non-negative |
| [`settime(spec)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#settime) | recompute `:time` from a column name or a per-row function; rows may only move later, and the result is re-clipped to `[start, stop)` |

The forward-looking counterparts — `lead`, and a permissive `settime` — live in
the `Acausal` submodule, under "Causality" below.

### Column transformations

The column set itself: which columns exist and in what order, and filling or
joining values into them. Rows pass through one for one.

| Operator | Semantics |
|---|---|
| [`selectcolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#selectcolumns) | keep the columns matching a name, `Regex`, name predicate, or collection of those (`:time` always kept) |
| [`dropcolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#dropcolumns) | drop the columns matching the same selector forms (`:time` never dropped) |
| [`reordercolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#reordercolumns) | move the matching columns to the front, in the selectors' order, the rest following in the input's order (`:time` always first) |
| [`forwardfill(sel...; key, tolerance)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#forwardfill) | replace `missing` in the selected columns with that column's last non-missing value, per key, while not older than `tolerance` |
| [`fillmissing(specs...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#fillmissing) | replace `missing` with a per-column constant, given as `name => value` pairs or a `NamedTuple` |
| [`asofjoin(right; key, tolerance, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#asofjoin) | append the most recent right-pipeline row at or before each row's time |

The forward-looking `futurejoin` lives in the `Acausal` submodule, under
"Causality" below.

### Summarizing transforms

Each folds one or more [summarizers](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/) over a set of rows.

| Operator | Semantics |
|---|---|
| [`summarize(ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarize) | summarize the whole window into rows at time `stop` |
| [`summarizecycles(ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarizecycles) | summarize each unique timestamp independently |
| [`intervalize(clock, ss; key, closelast)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#intervalize) | summarize over the intervals `[bₖ, bₖ₊₁)` a `clock` pipeline's times define, each emitted at its end time |
| [`summarizewindows(clock, lookback, ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarizewindows) | at each tick `τ` of a `clock` pipeline, summarize the trailing window `[τ - lookback, τ)`; one row per tick, or per key with rows in its window (plus one empty row when a key's window empties) |
| [`addsummarycolumns(ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#addsummarycolumns) | append running summary values after each row |
| [`addrollingcolumns(windows, ss; key, from)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#addrollingcolumns) | append summaries over named trailing windows, columns prefixed `{window}_` |

### Model fitting (MLJ)

| Operator | Semantics |
|---|---|
| [`applymodels(models; column, key, tolerance, strict, name, operation)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#applymodels) | append each row's prediction from the latest fitted model at or before it (per key), given a pipeline of `FitModel` outputs (needs `using MLJ`) |
| [`addpredictions(clock, lookback, model, predictors, response; key, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#addpredictions) | refit an MLJ model at every clock tick `τ` over `[τ - lookback, τ)`, per key, and append each row's prediction from the latest model — training rows always precede the rows they predict |
| [`modelreports(; column, name)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#modelreports) | over a pipeline of fitted models: replace each model with its fit report |

## Summarizers

The summarization transforms take one or more summarizers — or your own
`Summarizer` subtype — and an optional `key` (one or more column names) to
produce a separate summary per unique key value. Output columns are named by
suffix, shown here for a column `:x` (and `:y`, for the pairwise ones):

| Summarizer | Output | Structure | Semantics |
|---|---|---|---|
| [`Count()`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Count) | `:count` | group | rows summarized; `0` for no rows |
| [`CountDistinct(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#CountDistinct) | `:x_countdistinct` | monoid | distinct values seen, always an `Int`, `missing` counted as one of them; `0` for no rows |
| [`Sum(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Sum) | `:x_sum` | group | `Σx`, widening as `Base.sum` does; `0` for no rows |
| [`SumPower(:x, n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#SumPower) | `:x_sumpower_n` | group | `Σxⁿ`; `0` for no rows |
| [`Product(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Product) | `:x_product` | monoid | `Πx`; `1` for no rows |
| [`DotProduct(:x, :y)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#DotProduct) | `:x_y_dotproduct` | group | `Σxy`; `0` for no rows |
| [`Moment(:x, n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Moment) | `:x_moment_n` | group | the `n`-th raw moment, `Σxⁿ / count` |
| [`Mean(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Mean) | `:x_mean` | group | arithmetic mean |
| [`Variance(:x; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Variance) | `:x_variance` | group | variance, following `Statistics.var` |
| [`Std(:x; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Std) | `:x_std` | group | standard deviation, following `Statistics.std` |
| [`Covariance(:x, :y; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Covariance) | `:x_y_covariance` | group | covariance, following `Statistics.cov` |
| [`Correlation(:x, :y)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Correlation) | `:x_y_correlation` | group | Pearson correlation |
| [`LinearRegression(predictors, response; name, intercept)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#LinearRegression) | a block of columns under the `name` prefix | group | ordinary least squares; see below |
| [`Min(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Min) | `:x_min` | monoid | smallest value |
| [`Max(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Max) | `:x_max` | monoid | largest value |
| [`First(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#First) | `:x_first` | monoid | value of the earliest row |
| [`Last(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Last) | `:x_last` | monoid | value of the latest row |
| [`FitModel(model, predictors, response)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#FitModel) | `:model` | — | fits `model` to the rows it summarizes (needs `using MLJ`) |

Only the four accumulating summarizers and `CountDistinct` have an identity
element, given in the table; the rest summarize no rows as `missing`.
`CountDistinct` is also the one summarizer a `missing` does not poison — unlike
a sum, a distinct count stays knowable — and the one whose state is not O(1),
holding the distinct values it has seen. (MLJ exports a scientific type named
`Count`, so alongside `using MLJ` the summarizer is `CausalFrames.Count()`.)

```julia
p = readcsv("ticks.csv";
        types = Dict(:time => Int, :bid => Float64, :ask => Float64)) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]; key = :symbol)
```

`Moment`, `Mean`, `Variance`, `Std`, `Covariance`, `Correlation`, and
`LinearRegression` are *dependent* summarizers, computed from the counting and
accumulating ones — `Moment(:mid, n)` from `Count()` and `SumPower(:mid, n)`,
for instance. Those are folded alongside them but appear in the output only if
requested themselves.

`FitModel` emits the fitted model itself in a `:model` column — applied to a
stream by `applymodels`, refit on a rolling window by `addpredictions`, and
persisted by `writejls`.

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

Rolling windows, and the clock-sampled windows of `summarizewindows`, pick
their algorithm from the structure column above: `GroupSummarizer`s (`Sum`,
`Mean`, …) slide a running state in O(1) per row by subtracting exiting rows,
`MonoidSummarizer`s (`Min`, `Product`, …) fold each window from a segment tree
of partial combinations in O(log window), and summarizers declaring neither
re-fold each window from scratch — see DESIGN.md for the `combine!`/`downdate!`
interface a custom structured summarizer implements. A summarizer on one of
these paths should also implement `fresh!`, which zeroes a state in place: the
transforms zero a state tuple per cycle, per interval and per window query, so
building a new one there is a heap allocation per summarizer per row.

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

[`futurejoin`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#Acausal.futurejoin)
mirrors `asofjoin` with the match direction inverted — the **earliest** right
row whose time is not before the left row's, `missing` where none qualifies —
and [`lead`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#Acausal.lead)
mirrors `lag`; the permissive
[`settime`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#Acausal.settime)
mirrors the causal one. One cost worth knowing before
reaching for it: because `futurejoin` matches the earliest qualifying row, it
buffers right rows per key until a left row consumes or outruns them, and
proving that a key has no future match drains the right stream. Worst-case
memory is therefore O(right rows), against `asofjoin`'s O(keys).

The [Recipes](https://farrellm.github.io/CausalFrames.jl/dev/recipes/) page
collects the patterns that are not obvious from the operator list: ranking
within a group, joining a reference table, and using a derived stream as one.

See [DESIGN.md](DESIGN.md) for the full design.
