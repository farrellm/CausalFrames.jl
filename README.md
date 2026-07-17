# CausalFrames

[![Build Status](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/CausalFrames.jl/actions/workflows/CI.yml?query=branch%3Amaster)

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

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

## Summarizers

The summarization transforms take one or more summarizers — `Count()`,
`Sum(:col)`, `SumPower(:col, n)`, `Min(:col)`, `Max(:col)`, `First(:col)`,
`Last(:col)`, or your own `Summarizer` subtype — and an optional `key` (one or
more column names) to produce a separate summary per unique key value.
Output columns are named by suffix: `Sum(:mid)` produces `:mid_sum`,
`Min(:mid)` produces `:mid_min`, and `SumPower(:mid, 2)` produces `:mid_sum2`.

```julia
p = readcsv("ticks.csv") |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]; key = :symbol)
```

`Sum` and `SumPower` summarize no rows as `0`; `Min`, `Max`, `First`, and
`Last` have no identity element and yield `missing` instead.

An output column takes its element type from the input column: `Min`, `Max`,
`First`, and `Last` reproduce it verbatim, while `Sum` and `SumPower` widen it
exactly as `Base.sum` does (`Int32` sums to `Int64`, `Float32` to `Float32`).
Summarizers are typed from the input schema, so folding a large window
allocates on the order of kilobytes — see DESIGN.md for the interface a custom
`Summarizer` implements.

Every operator is **causal** — its output at time `t` depends only on input
rows with time `≤ t` — which is what makes streaming evaluation sound. See
[DESIGN.md](DESIGN.md) for the full design.
