# Recipes

Four patterns that are not obvious from the operator list. Each is a way of
saying something inside the pipeline that looks at first as though it needs an
exit to `DataFrame`.

## Ranking within a group

[`addsummarycolumns`](@ref) holds each summary *after* the current row has been
folded in, so a keyed [`Count`](@ref) is a 1-based ordinal within its key — the
nth row of that key so far:

```julia
p |> addsummarycolumns(Count(); key = :symbol)   # :count is 1, 2, 3, … per symbol
```

That ordinal counts in **stream order**, which is time order, so what it gives
you directly is `row_number() OVER (PARTITION BY symbol ORDER BY time)`. An
arbitrary `ORDER BY` is available wherever the row order within a group is free
— which is exactly when the partition column is `:time` itself, since rows
sharing a timestamp may arrive in any order. Sorting them at the source turns
the same count into a rank by whatever you sorted on:

```julia
# rows arrive sorted (year, votes descending), and :time is the year
readparquet("films.parquet"; time = :year) |>
    addsummarycolumns(Count(); key = :time) |>
    filterrows(r -> r.count <= 250)       # the 250 most-voted films of each year
```

The `ORDER BY` half is always the source's job. CausalFrames has no sort
operator, deliberately: monotone time is the invariant that makes streaming
sound, and a sort is not streaming. So the rank is only ever as meaningful as
the order the rows arrived in — which is worth asserting on if the source is not
under your control.

## Joining a reference table

[`asofjoin`](@ref) matches the most recent right row at or before each left
row's time. Give every right row the *same* time and that degenerates into a
plain keyed lookup: there is only one row per key, and it is always at or before
whatever the left row's time is. A dimension table, joined by key alone:

```julia
dim = readcsv("dim.csv"; types = Dict(:id => String), time = _ -> WINDOW_START)
facts |> asofjoin(dim; key = :id)
```

Memory is O(keys), as for any `asofjoin`.

The constant must fall inside the evaluation window. Sources clip to
`[start, stop)`, so a right pipeline timed outside it yields no rows at all —
and a right stream producing no chunks passes the left chunks through
*unchanged*, meaning the right columns are *absent from the schema* rather than
present and `missing`. The symptom is a confusing "column not found" from
whatever reads them next, at some distance from the cause. Time the reference
rows at the window's own `start`.

## Using a derived stream as a reference table

The recipe above needs constant-timed rows, and a stream that already carries
real times cannot be retimed backward: [`settime`](@ref) may only move rows
later. So to look up against something the pipeline itself derived, materialize
it and read it back at a constant time. [`writecsv`](@ref) is a pass-through
sink built for exactly this, and [`scan`](@ref) drives the pipeline for the side
effect alone:

```julia
derived |> selectcolumns(:id) |> writecsv(path) |> scan(ctx)

lookup = readcsv(path; types = Dict(:id => String), time = _ -> WINDOW_START)
facts |> asofjoin(lookup; key = :id)
```

Note that this is not a way around a pipeline being single-pass — it is not. A
`CausalPipeline` is a lazy description of how to produce data, and `load`,
`stream` and `scan` may each run it again; only the iterator produced by one
such run is single-pass. Materializing here buys the *retiming*, not the second
read.

If you would rather move the rows than write them out,
`CausalFrames.Acausal.settime` is the permissive form that may move rows
earlier. It lives in the `Acausal` submodule because that is a genuine
loss of causality, and it must be opted into by name.

## Fitting a model once and applying it later

[`addpredictions`](@ref) refits as the stream moves. To fit one model over a
training window and apply it to a later one, fit it with [`summarize`](@ref),
persist the one-row model table with [`writejls`](@ref) — CSV and parquet
cannot hold a fitted model — and read it back as the models pipeline of
[`applymodels`](@ref):

```julia
using MLJ                                   # loads the MLJ integration
model = (@load LinearRegressor pkg = MLJLinearModels)()

train = Context(DateTime(2024, 1, 1), DateTime(2025, 1, 1))
ticks |> summarize(FitModel(model, [:mid, :size], :ret)) |>
    writejls("model.jls") |> scan(train)

later = Context(DateTime(2026, 1, 1), DateTime(2027, 1, 1))
ticks |> applymodels(readjls("model.jls"); tolerance = Year(3)) |> load(later)
```

This is the reference-table problem again. `summarize` emits its row at the
training window's `stop`, 2025-01-01, and [`readjls`](@ref) clips to the window
it runs over like every source, so over `later` alone the model row is clipped
away and every prediction is `missing`. `tolerance` widens the models' window
backward far enough to reach it — and, being an as-of tolerance, also bounds
how old a model may be for each row. (Applied over a window that *starts* at
the training `stop`, the row is inside it and no tolerance is needed.)

The same file gives the fit's diagnostics: `readjls("model.jls") |>
modelreports()` over any window containing the model's time.
