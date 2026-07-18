# CausalFrames

[![Build Status](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://farrellm.github.io/CausalFrames.jl/dev/)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Time-series tables for Julia: DataFrames with a monotonically non-decreasing
`time` column, built lazily from composable pipelines.

```julia
using CausalFrames, Dates

p = readcsv("ticks.csv") |>
    filterrows(r -> r.price > 0) |>
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
  that does), `stream(ctx, pipeline)` yields frames incrementally.
- **`CausalFrame`** — a materialized table. It hides its backing storage
  (one or more time-disjoint chunks, wrapped without copying) and is
  accessed via the Tables.jl interface or `DataFrame(frame)`.

## Operators

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `clock(interval)` | source | one row per `interval` in `[start, stop)` |
| `readcsv(path)` | source | sorted CSV with a `time` column, clipped to `[start, stop)`, read incrementally |
| `filterrows(pred)` | transform | keep rows where `pred(row)` |
| `addcolumns(f)` | transform | `f(row)::NamedTuple` of new column values |
| `summarize(ss; key)` | transform | summarize the whole window into rows at time `stop` |
| `summarizecycles(ss; key)` | transform | summarize each unique timestamp independently |
| `addsummarycolumns(ss; key)` | transform | append running summary values after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | append summaries over named trailing windows, columns prefixed `{window}_` |
| `asofjoin(right; key, tolerance, ...)` | transform | append the most recent right-pipeline row at or before each row's time |

Each transform also has an uncurried, pipeline-first form — `filterrows(p, pred)`,
`addcolumns(p, f)`, `summarize(p, ss; key)` — equivalent to the `|>` chain
(`p |> filterrows(pred)`) for when the applied form reads clearer.

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

## Summarizers

The summarization transforms take one or more summarizers — `Count()`,
`Sum(:col)`, `SumPower(:col, n)`, `Moment(:col, n)`, `Min(:col)`,
`Max(:col)`, `First(:col)`, `Last(:col)`, or your own `Summarizer` subtype —
and an optional `key` (one or more column names) to produce a separate
summary per unique key value. Output columns are named by suffix:
`Sum(:mid)` produces `:mid_sum`, `Min(:mid)` produces `:mid_min`, and
`SumPower(:mid, 2)` produces `:mid_sumpower_2`.

```julia
p = readcsv("ticks.csv") |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]; key = :symbol)
```

`Sum` and `SumPower` summarize no rows as `0`; `Moment`, `Min`, `Max`,
`First`, and `Last` have no identity element and yield `missing` instead.
`Moment(:mid, n)` — the `n`-th raw moment, producing `:mid_moment_n` — is a
*dependent* summarizer, computed from `Count()` and `SumPower(:mid, n)`;
those are folded alongside it but appear in the output only if requested
themselves.

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
custom structured summarizer implements.

An output column takes its element type from the input column: `Min`, `Max`,
`First`, and `Last` reproduce it verbatim, while `Sum` and `SumPower` widen it
exactly as `Base.sum` does (`Int32` sums to `Int64`, `Float32` to `Float32`).
Summarizers are typed from the input schema, so folding a large window
allocates on the order of kilobytes — see DESIGN.md for the interface a custom
`Summarizer` implements.

Every operator is **causal** — its output at time `t` depends only on input
rows with time `≤ t` — which is what makes streaming evaluation sound. See
[DESIGN.md](DESIGN.md) for the full design.
