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
sharing a timestamp may arrive in any order. [`sortcycles`](@ref) orders them,
turning the same count into a rank by whatever you sorted on:

```julia
# :time is the year; within a year, votes descending, ties by id
readparquet("films.parquet"; time = :year) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :time) |>
    filterrows(r -> r.count <= 250)       # the 250 most-voted films of each year
```

The two halves of that `ORDER BY year, votes DESC, id` live in different places.
The time half belongs to the source — `sort = true` on [`readparquet`](@ref),
[`readcsv`](@ref) or [`readtable`](@ref) stably sorts a file or table not stored
in time order — because monotone time is the invariant that makes streaming
sound, and a sort across time cannot stream. The within-timestamp half can go
anywhere in the pipeline: `sortcycles` never moves a row in time, so it streams,
holding back only the latest cycle. Without it the rank is only as meaningful as
the order the rows arrived in.

## Joining a reference table

[`lookupjoin`](@ref) appends the columns of a table that has no time column — a
dimension table, a symbol master — to every row with the same key:

```julia
dim = DataFrame(id = ["a", "b"], sector = ["tech", "energy"])
facts |> lookupjoin(dim; key = :id)
```

Any Tables.jl table works, so a table kept on disk can be read with CSV.jl
directly, types inferred. It never passes through a source, so nothing clips it
to the window:

```julia
using CSV
facts |> lookupjoin(CSV.File("dim.csv"); key = :id)
```

A row whose key the table lacks gets `missing` in the new columns;
`unmatched = :drop` drops it instead, and `unmatched = :error` insists every key
is there. A table of keys alone with `unmatched = :drop` is a semi-join: it keeps
the rows whose key is listed.

A reference table that changes over time — a sector reassigned mid-window — is
not a lookup. Give its rows the times they take effect and use
[`asofjoin`](@ref), which picks the version in force at each row.

## Using a derived stream as a reference table

To look up against something a pipeline derived, materialize it and drop its
`:time` column, which `lookupjoin` refuses. A stream usually repeats its keys,
and a lookup table may not, so first decide which row stands for each key —
[`lastrow`](@ref) keeps the latest:

```julia
# yesterday's closing score per id, looked up by today's facts
closing = load(yesterday, scores |> lastrow(; key = :id))
facts |> lookupjoin(select(DataFrame(closing), Not(:time)); key = :id)
```

The table goes through a file as easily: [`writecsv`](@ref) is a pass-through
sink, [`scan`](@ref) drives the pipeline for that side effect alone, and
`CSV.File` can drop the time column as it reads:

```julia
scores |> lastrow(; key = :id) |> writecsv(path) |> scan(yesterday)
facts |> lookupjoin(CSV.File(path; drop = [:time]); key = :id)
```

Derive the table from an *earlier* window, as here. Built from the same window
as the facts, it would hand a row at time `t` a value computed from rows after
`t` — the lookup cannot know, since the table carries no times. Within one
window the causal form is an as-of join against the derived stream itself,
`facts |> asofjoin(scores; key = :id)`.

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
