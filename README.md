# CausalFrames

[![Build Status](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/farrellm/CausalFrames.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/farrellm/CausalFrames.jl)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://farrellm.github.io/CausalFrames.jl/dev/)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

Each name links to its [API reference](https://farrellm.github.io/CausalFrames.jl/dev/api/). Transforms are
curried for `|>`, and also take the pipeline first: `filterrows(p, pred)` is
`p |> filterrows(pred)`. Row functions receive a row supporting `row.time`,
`row.price` and `row[:price]`.

### Sources

| Operator | Semantics |
|---|---|
| [`emptyframe()`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#emptyframe) | no rows, only `:time` |
| [`concatenate(ps...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#concatenate) | the pipelines one after another, in time order, with identical columns |
| [`merge(ps...; batchsize)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#merge) | the pipelines interleaved by time, with the union of their columns (`missing` where one lacks a column); ties in argument order |
| [`clock(interval; batchsize)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#clock) | one row every `interval` in `[start, stop)` |
| [`readtable(table; time, checkorder, sort, closed, skipmissing)`](https://farrellm.github.io/CausalFrames.jl/dev/api/sources/#readtable) | an in-memory Tables.jl table or `CausalFrame`, clipped to the window |

### File I/O

Readers are sources and clip to `[start, stop)` (`closed = true` keeps rows at
`stop`), reading incrementally. Writers are pass-through transforms that write
each chunk as it flows by, on a background task.

| Operator | Semantics |
|---|---|
| [`readcsv(path; types, time, rename, delim, sort, chunkbytes, closed, skipmissing)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readcsv) | a CSV file; columns are `String` unless `types` says otherwise |
| [`readparquet(path; time, rename, sort, closed, skipmissing, backend)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readparquet) | a parquet file, skipping data outside the window |
| [`readjls(path; closed)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#readjls) | a file written by `writejls` |
| [`writecsv(path; queue, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writecsv) | write a CSV file |
| [`writeparquet(path; queue, rowgroupsize, backend, ...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writeparquet) | write a parquet file, valid once the stream is exhausted |
| [`writejls(path; queue)`](https://farrellm.github.io/CausalFrames.jl/dev/api/io/#writejls) | write with Julia's `Serialization`, for values CSV and parquet cannot hold (such as fitted models) |

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
| [`filterrows(pred)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#filterrows) | keep rows where `pred(row)` |
| [`addcolumns(f)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#addcolumns) | append the columns of the `NamedTuple` `f(row)` |
| [`head(n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#head) | the first `n` rows, then stop reading the input |
| [`lastrow(; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#lastrow) | the last row (per key), retimed to `stop` |
| [`sortcycles(by; rev)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#sortcycles) | stably sort the rows sharing each time |
| [`lag(offset)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#lag) | move every row `offset` later |
| [`settime(spec)`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#settime) | recompute `:time` from a column or function; rows may only move later |

### Column transformations

| Operator | Semantics |
|---|---|
| [`selectcolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#selectcolumns) | keep the matching columns (and `:time`) |
| [`dropcolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#dropcolumns) | drop the matching columns |
| [`reordercolumns(sel...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#reordercolumns) | move the matching columns to the front, after `:time` |
| [`forwardfill(sel...; key, tolerance)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#forwardfill) | fill `missing` with the last value (per key, within `tolerance`) |
| [`fillmissing(specs...)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#fillmissing) | fill `missing` with a constant per column |
| [`asofjoin(right; key, tolerance, strict, leftprefix, rightprefix, righttime)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#asofjoin) | append the latest `right` row at or before each row |
| [`lookupjoin(table; key, unmatched, leftprefix, rightprefix)`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#lookupjoin) | append the row with the same key from a table without time |

Column selectors are names, `Regex`es, name predicates, or collections of them.

### Summarizing transforms

Each folds one or more [summarizers](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/) over a set of rows; `key`
splits the fold by key value, and `keyset` declares the keys up front so every
key is emitted each time.

| Operator | Semantics |
|---|---|
| [`summarize(ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarize) | the whole window, emitted at `stop` |
| [`summarizecycles(ss; key, keyset)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarizecycles) | each time separately |
| [`intervalize(clock, ss; key, keyset, closelast)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#intervalize) | each interval between clock ticks, emitted at its end |
| [`summarizewindows(clock, lookback, ss; key, keyset)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#summarizewindows) | the window `[τ - lookback, τ)` at each clock tick `τ` |
| [`addsummarycolumns(ss; key)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#addsummarycolumns) | append running summaries of every row so far |
| [`addrollingcolumns(windows, ss; key, from)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizing/#addrollingcolumns) | append summaries of `[t - lookback, t]` for each named window |

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
| [`applymodels(models; column, key, tolerance, strict, name, operation)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#applymodels) | append each row's prediction from the latest model in `models` |
| [`addpredictions(clock, lookback, model, predictors, response; key, name, operation, verbosity)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#addpredictions) | refit `model` on a trailing window at each clock tick and append its predictions |
| [`modelreports(; column, name)`](https://farrellm.github.io/CausalFrames.jl/dev/api/models/#modelreports) | replace each fitted model with its fit report |

## Summarizers

Output columns are named by suffix, shown here for columns `:x` and `:y`.
Most summarize no rows as `missing`; the rest give the identity shown.

| Summarizer | Output | Structure | Semantics |
|---|---|---|---|
| [`Count()`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Count) | `:count` | group | number of rows; `0` |
| [`CountDistinct(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#CountDistinct) | `:x_countdistinct` | group | distinct values, `missing` included; `0` |
| [`Sum(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Sum) | `:x_sum` | group | `Σx`; `0` |
| [`SumPower(:x, n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#SumPower) | `:x_sumpower_n` | group | `Σxⁿ`; `0` |
| [`Product(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Product) | `:x_product` | monoid | `Πx`; `1` |
| [`DotProduct(:x, :y)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#DotProduct) | `:x_y_dotproduct` | group | `Σxy`; `0` |
| [`AgeWeightedSum(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#AgeWeightedSum) | `:x_ageweightedsum` | group | `Σk·x`, `k` the row's age (newest `0`); `0` |
| [`Moment(:x, n)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Moment) | `:x_moment_n` | group | `n`-th raw moment |
| [`Mean(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Mean) | `:x_mean` | group | mean |
| [`Variance(:x; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Variance) | `:x_variance` | group | variance, as `Statistics.var` |
| [`Std(:x; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Std) | `:x_std` | group | standard deviation, as `Statistics.std` |
| [`Covariance(:x, :y; corrected)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Covariance) | `:x_y_covariance` | group | covariance, as `Statistics.cov` |
| [`Correlation(:x, :y)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Correlation) | `:x_y_correlation` | group | Pearson correlation |
| [`LinearRegression(predictors, response; intercept, name)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#LinearRegression) | `n`, `r2`, `stderr`, and a beta and t statistic per term | group | ordinary least squares |
| [`Quantile(:x, p; interpolation)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Quantile) | `:x_quantile_50` for `p = 0.5`, one per `p` | group | quantiles, linear as `Statistics.quantile` or nearest-rank as TA-Lib's `PERCENTILE` |
| [`Median(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Median) | `:x_median` | group | median |
| [`PercentRank(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#PercentRank) | `:x_percentrank` | group | fraction of the other rows below the newest |
| [`CausalFrames.SortedValues(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#SortedValues) | `:x_sortedvalues` | group | the sorted values the three above read (unexported, for summarizers of your own) |
| [`Min(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Min) | `:x_min` | group | minimum |
| [`Max(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Max) | `:x_max` | group | maximum |
| [`First(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#First) | `:x_first` | group | value in the first row |
| [`Last(:x)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#Last) | `:x_last` | group | value in the last row |
| [`FitModel(model, predictors, response; name, verbosity)`](https://farrellm.github.io/CausalFrames.jl/dev/api/summarizers/#FitModel) | `:model` | — | a fitted MLJ model (needs `using MLJ`, where `Count` must be written `CausalFrames.Count`) |

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
`Base.sum` does.

The structure column sets each summarizer's window algorithm in
`addrollingcolumns` and `summarizewindows`: a group slides in O(1) per row, a
monoid folds a segment tree in O(log window), and anything else re-folds each
window. It is chosen per summarizer, so one slow summarizer does not slow the
rest of a call. See
[DESIGN.md](DESIGN.md) for the interface a custom summarizer implements.

## Causality

Every exported operator is **causal**: its output at time `t` depends only on
input rows with time `≤ t`. So loading `[a, c)` equals concatenating loads of
`[a, b)` and `[b, c)`, which makes streaming sound.

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

[`futurejoin`](https://farrellm.github.io/CausalFrames.jl/dev/api/columns/#Acausal.futurejoin) mirrors `asofjoin`, but can
buffer every right row; [`lead`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#Acausal.lead) mirrors `lag`; and
the permissive [`settime`](https://farrellm.github.io/CausalFrames.jl/dev/api/rows/#Acausal.settime), not exported even
from `Acausal`, mirrors `settime`.

The [Recipes](https://farrellm.github.io/CausalFrames.jl/dev/recipes/) page covers ranking within a group, reference-table
joins, and fitting a model once to apply later. See [DESIGN.md](DESIGN.md) for
the full design.
