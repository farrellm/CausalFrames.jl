# Recipes

Four patterns that are not obvious from the operator list, each of which might
seem to need an exit to `DataFrame`.

## Ranking within a group

[`addsummarycolumns`](@ref) includes the current row in each summary, so a keyed
[`Count`](@ref) numbers the rows of each key from 1:

```julia
p |> addsummarycolumns(Count(); key = :symbol)   # :count is 1, 2, 3, … per symbol
```

It counts in stream order, which is time order: SQL's
`row_number() OVER (PARTITION BY symbol ORDER BY time)`. To rank by something
else, partition by `:time` itself, whose rows may arrive in any order, and order
them with [`sortcycles`](@ref):

```julia
# :time is the year; within a year, votes descending, ties by id
readparquet("films.parquet"; time = :year) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :time) |>
    filterrows(r -> r.count <= 250)       # the 250 most-voted films of each year
```

The time half of that `ORDER BY year, votes DESC, id` belongs to the source:
`sort = true` on [`readparquet`](@ref), [`readcsv`](@ref) or
[`readtable`](@ref) sorts data not stored in time order, since a sort across
time cannot stream. The within-time half can go anywhere, because `sortcycles`
never moves a row in time.

## Joining a reference table

[`lookupjoin`](@ref) appends the columns of a table without time, such as a
dimension table, to every row with the same key:

```julia
dim = DataFrame(id = ["a", "b"], sector = ["tech", "energy"])
facts |> lookupjoin(dim; key = :id)
```

Any Tables.jl table works, so a table on disk can be read with CSV.jl directly,
with inferred types:

```julia
using CSV
facts |> lookupjoin(CSV.File("dim.csv"); key = :id)
```

A row whose key is missing from the table gets `missing`; `unmatched = :drop`
drops it and `unmatched = :error` throws. A table of keys alone with
`unmatched = :drop` is a semi-join.

A reference table that changes over time is not a lookup: give its rows the
times they take effect and use [`asofjoin`](@ref).

## Using a derived stream as a reference table

To look up against a pipeline's output, load it and drop its `:time` column.
A lookup table has one row per key, so pick a row per key first;
[`lastrow`](@ref) keeps the latest:

```julia
# yesterday's closing score per id, looked up by today's facts
closing = load(yesterday, scores |> lastrow(; key = :id))
facts |> lookupjoin(select(DataFrame(closing), Not(:time)); key = :id)
```

Or go through a file, with [`writecsv`](@ref) and [`scan`](@ref), letting
`CSV.File` drop the time column:

```julia
scores |> lastrow(; key = :id) |> writecsv(path) |> scan(yesterday)
facts |> lookupjoin(CSV.File(path; drop = [:time]); key = :id)
```

Derive the table from an *earlier* window, as here. From the same window, it
would give a row at time `t` values computed from rows after `t`. Within one
window, join the derived stream itself instead:
`facts |> asofjoin(scores; key = :id)`.

## Fitting a model once and applying it later

[`addpredictions`](@ref) refits as the stream moves. To fit one model on a
training window and apply it to a later one, fit it with [`summarize`](@ref),
save it with [`writejls`](@ref) (CSV and parquet cannot hold a model), and pass
it to [`applymodels`](@ref):

```julia
using MLJ                                   # loads the MLJ integration
model = (@load LinearRegressor pkg = MLJLinearModels)()

train = Context(DateTime(2024, 1, 1), DateTime(2025, 1, 1))
ticks |> summarize(FitModel(model, [:mid, :size], :ret)) |>
    writejls("model.jls") |> scan(train)

later = Context(DateTime(2026, 1, 1), DateTime(2027, 1, 1))
ticks |> applymodels(readjls("model.jls"); tolerance = Year(3)) |> load(later)
```

The model row is timed at the training window's `stop`, 2025-01-01, and
[`readjls`](@ref) clips to the window it runs over, so over `later` alone every
prediction would be `missing`. `tolerance` widens the models' window back far
enough to reach the row, and also caps how old a model may be. A window starting
at the training `stop` needs no tolerance.

`readjls("model.jls") |> modelreports()`, over a window containing the model's
time, gives the fit's diagnostics.
