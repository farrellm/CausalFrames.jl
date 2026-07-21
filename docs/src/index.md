# CausalFrames.jl

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
  that does), `stream(ctx, pipeline)` yields frames incrementally.
- **`CausalFrame`** — a materialized table. It hides its backing storage
  (one or more time-disjoint chunks, wrapped without copying) and is
  accessed via the Tables.jl interface or `DataFrame(frame)`.

## Operators

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `clock(interval)` | source | one row per `interval` in `[start, stop)` |
| `readcsv(path; types, time, rename, delim)` | source | CSV read as `String` columns (`types` opts columns into concrete types); `time` picks the time column by name or a per-row function; clipped to `[start, stop)`, read incrementally |
| `filterrows(pred)` | transform | keep rows where `pred(row)` |
| `addcolumns(f)` | transform | `f(row)::NamedTuple` of new column values |
| `selectcolumns(sel...)` | transform | keep the columns matching a name, `Regex`, name predicate, or collection of those (`:time` always kept) |
| `dropcolumns(sel...)` | transform | drop the columns matching the same selector forms (`:time` never dropped) |
| `summarize(ss; key)` | transform | summarize the whole window into rows at time `stop` |
| `summarizecycles(ss; key)` | transform | summarize each unique timestamp independently |
| `addsummarycolumns(ss; key)` | transform | append running summary values after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | append summaries over named trailing windows, columns prefixed `{window}_` |
| `asofjoin(right; key, tolerance, strict, ...)` | transform | append the most recent right row not after each left row, per key; `missing` where none |

Row functions receive a map-like row object: `row.time`, `row.price`,
`row[:price]`.

## Summarizers

The summarization transforms take one or more summarizers — `Count()`,
`Sum(:col)`, `SumPower(:col, n)`, `Product(:col)`, `DotProduct(:a, :b)`,
`Moment(:col, n)`, `Mean(:col)`, `Variance(:col)`, `Std(:col)`,
`Covariance(:a, :b)`, `Correlation(:a, :b)`, `Min(:col)`, `Max(:col)`,
`First(:col)`, `Last(:col)`,
or your own `Summarizer` subtype — and an optional `key` (one or more column
names) to produce a separate summary per unique key value. Output columns are
named by suffix: `Sum(:mid)` produces `:mid_sum`, `Min(:mid)` produces
`:mid_min`, `SumPower(:mid, 2)` produces `:mid_sumpower_2`, and the two-column
`DotProduct(:bid, :ask)` produces `:bid_ask_dotproduct`.

```julia
p = readcsv("ticks.csv";
        types = Dict(:time => Int, :bid => Float64, :ask => Float64)) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    addsummarycolumns([Count(), Sum(:mid), Min(:mid), Max(:mid)]; key = :symbol)
```

`Sum`, `SumPower`, `Product`, and `DotProduct` have an identity element, so
they summarize no rows as `0` (`Product` as `1`); the rest have none and yield
`missing` instead. `Moment(:mid, n)` — the `n`-th raw moment, producing
`:mid_moment_n` — is a *dependent* summarizer, computed from `Count()` and
`SumPower(:mid, n)`; `Mean`, `Variance`, `Std`, `Covariance`, and
`Correlation` are dependent too. Dependencies are folded alongside a
summarizer but appear in the output only if requested themselves. `Variance`,
`Std`, and `Covariance` follow `Statistics`, taking a `corrected::Bool = true`
keyword (the divisor is `n - Int(corrected)`); `Correlation` follows
`Statistics.cor`, taking no `corrected` keyword and clamping to `[-1, 1]`.

An output column takes its element type from the input column: `Min`, `Max`,
`First`, and `Last` reproduce it verbatim, while `Sum` and `SumPower` widen it
exactly as `Base.sum` does (`Int32` sums to `Int64`, `Float32` to `Float32`).
Summarizers are typed from the input schema, so folding a large window
allocates on the order of kilobytes.

Every operator is **causal** — its output at time `t` depends only on input
rows with time `≤ t` — which is what makes streaming evaluation sound. See
`DESIGN.md` in the repository for the full design, including the interface a
custom `Summarizer` implements.
