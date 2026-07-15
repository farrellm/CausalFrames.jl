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
  conceptually a function from a `Context` to a frame. Built from sources
  and chained with `|>`.
- **`CausalFrame`** — the materialized result of `load(ctx, pipeline)`. It
  hides its backing storage (one or more time-disjoint chunks, the basis
  for streaming) and is accessed via the Tables.jl interface or
  `DataFrame(frame)`.

## Operators

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `clock(interval)` | source | one row per `interval` in `[start, stop)` |
| `readcsv(path)` | source | sorted CSV with a `time` column, clipped to `[start, stop)` |
| `filterrows(pred)` | transform | keep rows where `pred(row)` |
| `addcolumns(f)` | transform | `f(row)::NamedTuple` of new column values |

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

Every operator is **causal** — its output at time `t` depends only on input
rows with time `≤ t` — which is what makes chunked and (future) streaming
evaluation sound. See [DESIGN.md](DESIGN.md) for the full design.
