---
name: causalframes
description: >-
  How to use CausalFrames.jl, the Julia package for time-series tables built
  from lazy `|>` pipelines: CSV/parquet sources, filterrows, addcolumns,
  asofjoin, forwardfill, rolling and clock-windowed summaries, bars, and MLJ
  model fitting. Use this skill whenever Julia code does or should do
  `using CausalFrames`, or the user wants time-series, tick or bar data
  processed in Julia (resampling, rolling statistics, as-of joins, OHLC bars,
  point-in-time features free of lookahead), even if they don't name the
  package.
---
<!-- Generated from the CausalFrames.jl docs by docs/skill.jl; edit README.md,
     docs/src and the docstrings instead, then run docs/make.jl. -->

# CausalFrames.jl

Time-series tables for Julia: DataFrames with a monotonically non-decreasing
`time` column, built lazily from composable pipelines.

```julia
using CausalFrames, Dates

path = joinpath(mktempdir(), "ticks.csv")
write(path, """
    time,bid,ask
    2026-01-02T09:30:00,100.0,100.2
    2026-01-02T09:31:00,-1.0,100.3
    2026-01-02T09:32:00,100.4,100.6
    """)

p = readcsv(path; types = Dict(:time => DateTime, :bid => Float64, :ask => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

ctx = Context(DateTime(2026, 1, 1), DateTime(2026, 2, 1))
load(ctx, p)

# output

CausalFrame{Dates.DateTime} with 2 rows over [2026-01-01T00:00:00, 2026-02-01T00:00:00]
 Row │ time                 bid      ask      mid
     │ DateTime             Float64  Float64  Float64
─────┼────────────────────────────────────────────────
   1 │ 2026-01-02T09:30:00    100.0    100.2    100.1
   2 │ 2026-01-02T09:32:00    100.4    100.6    100.5
```

## Using this skill

This page is the package README: the concepts, every operator and summarizer
in one table, and worked examples. Each name links to its full docstring in
`references/` (signature, every argument and keyword with its default and
constraints, and an example with its output), and `references/recipes.md`
covers patterns that aren't obvious from the tables. Read an operator's
docstring before relying on its keywords; the details (what `key` accepts, how
`tolerance` widens the input, which output columns appear) are there, not here.

- Stay in the pipeline. Build with `|>` and materialize once, with `load(ctx,
  p)` or `DataFrame(load(ctx, p))`. Most things that seem to need a detour
  through DataFrames have a pipeline form; check the recipes before leaving.
- Causality is the point of the package: an operator's output at time `t`
  uses only rows at or before `t`, so features computed with it carry no
  lookahead. Reach for `CausalFrames.Acausal` only when forward-looking data
  is the goal (labels, targets), and say so to the user.
- Sources clip to the context's `[start, stop)`, and `:time` must be
  non-decreasing; the readers' `sort` keyword handles unsorted input.
- Examples print their result after `# output`. That is the expected output,
  not code to run.
- Check a pipeline by running it on a few rows from
  `readtable(DataFrame(...))` and comparing against what you expect.

## Concepts

- **`Context(start, stop)`**: a time window. The time type can be anything
  ordered: `DateTime`, `Int` ticks, `Float64` seconds.
- **`CausalPipeline`**: a lazy recipe for producing data over a context, built
  from a source and chained with `|>`. It runs only when evaluated:
  `load(ctx, p)` materializes the window, `stream(ctx, p)` yields it chunk by
  chunk, and `scan(ctx, p)` runs it for its side effects. Each also has a
  curried form, `p |> load(ctx)`.
- **`CausalFrame`**: a loaded table. Read it through Tables.jl or copy it out
  with `DataFrame(frame)`.

## Operators

Each name links to its [API reference](references/). Transforms are
curried for `|>`, and also take the pipeline first: `filterrows(p, pred)` is
`p |> filterrows(pred)`. Row functions receive a row supporting `row.time`,
`row.price` and `row[:price]`.

### Sources

| Operator | Semantics |
|---|---|
| [`emptyframe()`](references/sources.md#emptyframe) | no rows, only `:time` |
| [`concatenate(ps...)`](references/sources.md#concatenate) | the pipelines one after another, in time order, with identical columns |
| [`merge(ps...; batchsize)`](references/sources.md#merge) | the pipelines interleaved by time, with the union of their columns (`missing` where one lacks a column); ties in argument order |
| [`clock(interval; batchsize)`](references/sources.md#clock) | one row every `interval` in `[start, stop)` |
| [`readtable(table; time, checkorder, sort, closed, skipmissing)`](references/sources.md#readtable) | an in-memory Tables.jl table or `CausalFrame`, clipped to the window |

### File I/O

Readers are sources and clip to `[start, stop)` (`closed = true` keeps rows at
`stop`), reading incrementally. Writers are pass-through transforms that write
each chunk as it flows by, on a background task.

| Operator | Semantics |
|---|---|
| [`readcsv(path; types, time, rename, delim, sort, chunkbytes, closed, skipmissing)`](references/io.md#readcsv) | a CSV file; columns are `String` unless `types` says otherwise |
| [`readparquet(path; time, rename, sort, closed, skipmissing, backend)`](references/io.md#readparquet) | a parquet file, skipping data outside the window |
| [`readjls(path; closed)`](references/io.md#readjls) | a file written by `writejls` |
| [`writecsv(path; queue, ...)`](references/io.md#writecsv) | write a CSV file |
| [`writeparquet(path; queue, rowgroupsize, backend, ...)`](references/io.md#writeparquet) | write a parquet file, valid once the stream is exhausted |
| [`writejls(path; queue)`](references/io.md#writejls) | write with Julia's `Serialization`, for values CSV and parquet cannot hold (such as fitted models) |

Parquet needs a backend: `using DuckDB` or `using Parquet2` enables both
operators. Reading prefers DuckDB and writing Parquet2; `backend` overrides.

End a chain with `scan` to run it for the file alone:

```julia
clean = joinpath(mktempdir(), "clean.csv")
p |> writecsv(clean) |> scan(ctx)
print(read(clean, String))

# output

time,bid,ask,mid
2026-01-02T09:30:00.0,100.0,100.2,100.1
2026-01-02T09:32:00.0,100.4,100.6,100.5
```

### Row transformations

| Operator | Semantics |
|---|---|
| [`filterrows(pred)`](references/rows.md#filterrows) | keep rows where `pred(row)` |
| [`addcolumns(f)`](references/rows.md#addcolumns) | append the columns of the `NamedTuple` `f(row)` |
| [`head(n)`](references/rows.md#head) | the first `n` rows, then stop reading the input |
| [`lastrow(; key)`](references/rows.md#lastrow) | the last row (per key), retimed to `stop` |
| [`sortcycles(by; rev)`](references/rows.md#sortcycles) | stably sort the rows sharing each time |
| [`lag(offset)`](references/rows.md#lag) | move every row `offset` later |
| [`warmup(lookback, f)`](references/rows.md#warmup) | run `f` from `lookback` before the window, so its state starts warm |
| [`settime(spec)`](references/rows.md#settime) | recompute `:time` from a column or function; rows may only move later |

### Column transformations

| Operator | Semantics |
|---|---|
| [`selectcolumns(sel...)`](references/columns.md#selectcolumns) | keep the matching columns (and `:time`) |
| [`dropcolumns(sel...)`](references/columns.md#dropcolumns) | drop the matching columns |
| [`reordercolumns(sel...)`](references/columns.md#reordercolumns) | move the matching columns to the front, after `:time` |
| [`forwardfill(sel...; key, tolerance)`](references/columns.md#forwardfill) | fill `missing` with the last value (per key, within `tolerance`) |
| [`fillmissing(specs...)`](references/columns.md#fillmissing) | fill `missing` with a constant per column |
| [`asofjoin(right; key, tolerance, strict, leftprefix, rightprefix, righttime)`](references/columns.md#asofjoin) | append the latest `right` row at or before each row |
| [`lookupjoin(table; key, unmatched, leftprefix, rightprefix)`](references/columns.md#lookupjoin) | append the row with the same key from a table without time |

Column selectors are names, `Regex`es, name predicates, or collections of them.

### Summarizing transforms

Each folds one or more [summarizers](references/summarizers.md) over a set of rows; `key`
splits the fold by key value, and `keyset` declares the keys up front so every
key is emitted each time.

| Operator | Semantics |
|---|---|
| [`summarize(ss; key)`](references/summarizing.md#summarize) | the whole window, emitted at `stop` |
| [`summarizecycles(ss; key, keyset)`](references/summarizing.md#summarizecycles) | each time separately |
| [`intervalize(clock, ss; key, keyset, closelast)`](references/summarizing.md#intervalize) | each interval between clock ticks, emitted at its end |
| [`summarizewindows(clock, lookback, ss; key, keyset)`](references/summarizing.md#summarizewindows) | the window `[τ - lookback, τ)` at each clock tick `τ` |
| [`addsummarycolumns(ss; key)`](references/summarizing.md#addsummarycolumns) | append running summaries of every row so far |
| [`addrollingcolumns(windows, ss; key, from, sharedrun)`](references/summarizing.md#addrollingcolumns) | append summaries of `[t - lookback, t]`, or of the last `n` rows under [`Bars(n)`](references/summarizing.md#Bars), for each named window |

```julia
p |> addrollingcolumns((m1 = Minute(1), h1 = Hour(1)), Mean(:mid)) |>
    selectcolumns(:mid, r"_mean") |>
    load(ctx)

# output

CausalFrame{Dates.DateTime} with 2 rows over [2026-01-01T00:00:00, 2026-02-01T00:00:00]
 Row │ time                 mid      m1_mid_mean  h1_mid_mean
     │ DateTime             Float64  Float64?     Float64?
─────┼────────────────────────────────────────────────────────
   1 │ 2026-01-02T09:30:00    100.1        100.1        100.1
   2 │ 2026-01-02T09:32:00    100.5        100.5        100.3
```

### Model fitting (MLJ)

These need `using MLJ`.

| Operator | Semantics |
|---|---|
| [`applymodels(models; column, key, tolerance, strict, name, operation)`](references/models.md#applymodels) | append each row's prediction from the latest model in `models` |
| [`addpredictions(clock, lookback, model, predictors, response; key, name, operation, verbosity)`](references/models.md#addpredictions) | refit `model` on a trailing window at each clock tick and append its predictions |
| [`modelreports(; column, name)`](references/models.md#modelreports) | replace each fitted model with its fit report |

## Summarizers

Output columns are named by suffix, shown here for columns `:x` and `:y`.
Most summarize no rows as `missing`; the rest give the identity shown. Every
summarizer but `FitModel` also takes a [row term](references/summarizers.md)
`name => f` in place of a column, reading `f(row)` as a virtual column `name`
that never reaches the output: `Sum(:range => r -> r.high - r.low)` produces
`:range_sum`.

| Summarizer | Output | Structure | Semantics |
|---|---|---|---|
| [`Count()`](references/summarizers.md#Count) | `:count` | group | number of rows; `0` |
| [`CountDistinct(:x)`](references/summarizers.md#CountDistinct) | `:x_countdistinct` | group | distinct values, `missing` included; `0` |
| [`Sum(:x)`](references/summarizers.md#Sum) | `:x_sum` | group | `Σx`; `0` |
| [`SumPower(:x, n)`](references/summarizers.md#SumPower) | `:x_sumpower_n` | group | `Σxⁿ`; `0` |
| [`Product(:x)`](references/summarizers.md#Product) | `:x_product` | monoid | `Πx`; `1` |
| [`DotProduct(:x, :y)`](references/summarizers.md#DotProduct) | `:x_y_dotproduct` | group | `Σxy`; `0` |
| [`AgeWeightedSum(:x)`](references/summarizers.md#AgeWeightedSum) | `:x_ageweightedsum` | group | `Σk·x`, `k` the row's age (newest `0`); `0` |
| [`Moment(:x, n)`](references/summarizers.md#Moment) | `:x_moment_n` | group | `n`-th raw moment |
| [`Mean(:x)`](references/summarizers.md#Mean) | `:x_mean` | group | mean |
| [`Variance(:x; corrected)`](references/summarizers.md#Variance) | `:x_variance` | group | variance, as `Statistics.var` |
| [`Std(:x; corrected)`](references/summarizers.md#Std) | `:x_std` | group | standard deviation, as `Statistics.std` |
| [`Covariance(:x, :y; corrected)`](references/summarizers.md#Covariance) | `:x_y_covariance` | group | covariance, as `Statistics.cov` |
| [`Correlation(:x, :y)`](references/summarizers.md#Correlation) | `:x_y_correlation` | group | Pearson correlation |
| [`LinearRegression(predictors, response; intercept, name)`](references/summarizers.md#LinearRegression) | `n`, `r2`, `stderr`, and a beta and t statistic per term | group | ordinary least squares |
| [`Quantile(:x, p; interpolation)`](references/summarizers.md#Quantile) | `:x_quantile_50` for `p = 0.5`, one per `p` | group | quantiles, linear as `Statistics.quantile` or nearest-rank as TA-Lib's `PERCENTILE` |
| [`Median(:x)`](references/summarizers.md#Median) | `:x_median` | group | median |
| [`PercentRank(:x)`](references/summarizers.md#PercentRank) | `:x_percentrank` | group | fraction of the other rows below the newest |
| [`MeanAbsDev(:x)`](references/summarizers.md#MeanAbsDev) | `:x_meanabsdev` | group | mean absolute deviation about the mean |
| [`CausalFrames.SortedValues(:x)`](references/summarizers.md#SortedValues) | `:x_sortedvalues` | group | the sorted values the three above read (unexported, for summarizers of your own) |
| [`CausalFrames.WindowValues(:x)`](references/summarizers.md#WindowValues) | `:x_windowvalues` | group | the values in arrival order `MeanAbsDev` reads (unexported, for summarizers of your own) |
| [`Min(:x)`](references/summarizers.md#Min) | `:x_min` | group | minimum |
| [`Max(:x)`](references/summarizers.md#Max) | `:x_max` | group | maximum |
| [`MinIndex(:x)`](references/summarizers.md#MinIndex) | `:x_minindex` | group | rows since the minimum (`Int`; ties go to the newest) |
| [`MaxIndex(:x)`](references/summarizers.md#MaxIndex) | `:x_maxindex` | group | rows since the maximum (`Int`; ties go to the newest) |
| [`MinWithIndex(:x)`](references/summarizers.md#MinWithIndex) | `:x_min`, `:x_minindex` | group | both, from one state |
| [`MaxWithIndex(:x)`](references/summarizers.md#MaxWithIndex) | `:x_max`, `:x_maxindex` | group | both, from one state |
| [`First(:x)`](references/summarizers.md#First) | `:x_first` | group | value in the first row |
| [`Last(:x)`](references/summarizers.md#Last) | `:x_last` | group | value in the last row |
| [`FitModel(model, predictors, response; name, verbosity)`](references/summarizers.md#FitModel) | `:model` | — | a fitted MLJ model (needs `using MLJ`, where `Count` must be written `CausalFrames.Count`) |

```julia
p |> addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]) |>
    selectcolumns(:count, r"mid_") |>
    load(ctx)

# output

CausalFrame{Dates.DateTime} with 2 rows over [2026-01-01T00:00:00, 2026-02-01T00:00:00]
 Row │ time                 count  mid_sum  mid_min  mid_max
     │ DateTime             Int64  Float64  Float64  Float64
─────┼───────────────────────────────────────────────────────
   1 │ 2026-01-02T09:30:00      1    100.1    100.1    100.1
   2 │ 2026-01-02T09:32:00      2    200.6    100.1    100.5
```

Moments, variances, correlations and regressions are computed from shared
counts and sums, which are folded once however many summarizers need them and
appear in the output only if requested. An output column takes its element type
from the input: `Min`, `Max`, `First` and `Last` keep it, and `Sum` widens as
`Base.sum` does. Counts and row indices are `Int`.

The structure column sets each summarizer's window algorithm in
`addrollingcolumns` and `summarizewindows`: a group slides in O(1) per row, a
monoid folds a segment tree in O(log window), and anything else re-folds each
window. It is chosen per summarizer, so one slow summarizer does not slow the
rest of a call. See
[DESIGN.md](https://github.com/farrellm/CausalFrames.jl/blob/master/DESIGN.md) for the interface a custom summarizer implements.

## Causality

Every exported operator is **causal**: its output at time `t` depends only on
input rows with time `≤ t`. So streaming a pipeline chunk by chunk gives the
same rows as loading it, and for sources and row-wise transforms, loading
`[a, c)` equals concatenating loads of `[a, b)` and `[b, c)`.

The forward-looking exceptions live in the `Acausal` submodule, which is never
re-exported:

```julia
using CausalFrames.Acausal, DataFrames

quotes = readtable(DataFrame(time = [1, 3], bid = [10.0, 10.2]))
fills = readtable(DataFrame(time = [2, 3], qty = [5, 7]))

# the earliest fill at or after each quote
quotes |> futurejoin(fills; righttime = :filltime) |> load(Context(0, 10))

# output

CausalFrame{Int64} with 2 rows over [0, 10]
 Row │ time   bid      qty     filltime
     │ Int64  Float64  Int64?  Int64?
─────┼──────────────────────────────────
   1 │     1     10.0       5         2
   2 │     3     10.2       7         3
```

[`futurejoin`](references/columns.md#Acausal.futurejoin) mirrors `asofjoin`, but can
buffer every right row; [`lead`](references/rows.md#Acausal.lead) mirrors `lag`; and
the permissive [`settime`](references/rows.md#Acausal.settime), not exported even
from `Acausal`, mirrors `settime`.

The [Recipes](references/recipes.md) page covers ranking within a group, reference-table
joins, and fitting a model once to apply later. See [DESIGN.md](https://github.com/farrellm/CausalFrames.jl/blob/master/DESIGN.md) for
the full design.

Coding agents can learn the package from the
[CausalFrames skill](https://github.com/farrellm/CausalFrames.jl/tree/master/skills/causalframes),
generated from these docs: copy that directory into a project's
`.claude/skills/` (or `~/.claude/skills/`).
