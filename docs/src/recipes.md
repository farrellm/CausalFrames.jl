# Recipes

```@meta
DocTestSetup = :(using CausalFrames, DataFrames, Dates)
```

Four patterns that are not obvious from the operator list, each of which might
seem to need an exit to `DataFrame`.

## Ranking within a group

[`addsummarycolumns`](@ref) includes the current row in each summary, so a keyed
[`Count`](@ref) numbers the rows of each key from 1:

```jldoctest
p = readtable(DataFrame(time = [1, 1, 2, 3], symbol = ["a", "b", "a", "a"]))
p |> addsummarycolumns(Count(); key = :symbol) |> load(Context(0, 10))

# output

CausalFrame{Int64} with 4 rows over [0, 10]
 Row │ time   symbol  count
     │ Int64  String  Int64
─────┼──────────────────────
   1 │     1  a           1
   2 │     1  b           1
   3 │     2  a           2
   4 │     3  a           3
```

It counts in stream order, which is time order: SQL's
`row_number() OVER (PARTITION BY symbol ORDER BY time)`. To rank by something
else, partition by `:time` itself, whose rows may arrive in any order, and order
them with [`sortcycles`](@ref):

```jldoctest
films = DataFrame(year = [2021, 2020, 2020, 2020, 2021],
                  id = [4, 1, 2, 3, 5], votes = [1, 5, 9, 9, 7])

# :time is the year; within a year, votes descending, ties by id
readtable(films; time = :year, sort = true) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :time) |>
    filterrows(r -> r.count <= 2) |>      # the 2 most-voted films of each year
    load(Context(2020, 2030))

# output

CausalFrame{Int64} with 4 rows over [2020, 2030]
 Row │ time   id     votes  count
     │ Int64  Int64  Int64  Int64
─────┼────────────────────────────
   1 │  2020      2      9      1
   2 │  2020      3      9      2
   3 │  2021      5      7      1
   4 │  2021      4      1      2
```

The time half of that `ORDER BY year, votes DESC, id` belongs to the source:
`sort = true` on [`readparquet`](@ref), [`readcsv`](@ref) or
[`readtable`](@ref) sorts data not stored in time order, since a sort across
time cannot stream. The within-time half can go anywhere, because `sortcycles`
never moves a row in time.

## Joining a reference table

[`lookupjoin`](@ref) appends the columns of a table without time, such as a
dimension table, to every row with the same key:

```jldoctest lookup
facts = readtable(DataFrame(time = [1, 2, 3], id = ["a", "b", "c"], qty = [5, 6, 7]))
dim = DataFrame(id = ["a", "b"], sector = ["tech", "energy"])
facts |> lookupjoin(dim; key = :id) |> load(Context(0, 10))

# output

CausalFrame{Int64} with 3 rows over [0, 10]
 Row │ time   id      qty    sector
     │ Int64  String  Int64  String?
─────┼───────────────────────────────
   1 │     1  a           5  tech
   2 │     2  b           6  energy
   3 │     3  c           7  missing
```

Any Tables.jl table works, so a table on disk can be read with CSV.jl directly,
with inferred types:

```jldoctest lookup
using CSV
path = joinpath(mktempdir(), "dim.csv")
CSV.write(path, dim)
facts |> lookupjoin(CSV.File(path); key = :id) |> load(Context(0, 10))

# output

CausalFrame{Int64} with 3 rows over [0, 10]
 Row │ time   id      qty    sector
     │ Int64  String  Int64  String7?
─────┼────────────────────────────────
   1 │     1  a           5  tech
   2 │     2  b           6  energy
   3 │     3  c           7  missing
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

```jldoctest derived
scores = readtable(DataFrame(time = [1, 5, 8, 12], id = ["a", "b", "a", "a"],
                             score = [0.1, 0.2, 0.3, 0.9]))
facts = readtable(DataFrame(time = [11, 13], id = ["a", "b"]))
yesterday, today = Context(0, 10), Context(10, 20)

# yesterday's closing score per id, looked up by today's facts
closing = load(yesterday, scores |> lastrow(; key = :id))
facts |> lookupjoin(select(DataFrame(closing), Not(:time)); key = :id) |> load(today)

# output

CausalFrame{Int64} with 2 rows over [10, 20]
 Row │ time   id      score
     │ Int64  String  Float64?
─────┼─────────────────────────
   1 │    11  a            0.3
   2 │    13  b            0.2
```

Or go through a file, with [`writecsv`](@ref) and [`scan`](@ref), letting
`CSV.File` drop the time column:

```jldoctest derived
using CSV
path = joinpath(mktempdir(), "closing.csv")
scores |> lastrow(; key = :id) |> writecsv(path) |> scan(yesterday)
facts |> lookupjoin(CSV.File(path; drop = [:time]); key = :id) |> load(today)

# output

CausalFrame{Int64} with 2 rows over [10, 20]
 Row │ time   id      score
     │ Int64  String  Float64?
─────┼─────────────────────────
   1 │    11  a            0.3
   2 │    13  b            0.2
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

```jldoctest model
using MLJBase, MLJLinearModels   # usually just `using MLJ`
model = MLJLinearModels.LinearRegressor()

days = Date(2024, 1, 1):Day(1):Date(2026, 12, 31)
x = [sin(i) for i in eachindex(days)]
ticks = readtable(DataFrame(time = collect(days), x = x, y = 2 .* x .+ 1))

path = joinpath(mktempdir(), "model.jls")
train = Context(Date(2024, 1, 1), Date(2025, 1, 1))
ticks |> summarize(FitModel(model, :x, :y)) |> writejls(path) |> scan(train)

later = Context(Date(2026, 1, 1), Date(2027, 1, 1))
ticks |> applymodels(readjls(path); tolerance = Day(3 * 365)) |>
    addcolumns(r -> (; rounded = round(r.prediction; digits = 3))) |>
    selectcolumns(:y, :rounded) |> head(3) |> load(later)

# output

CausalFrame{Dates.Date} with 3 rows over [2026-01-01, 2027-01-01]
 Row │ time        y          rounded
     │ Date        Float64    Float64
─────┼────────────────────────────────
   1 │ 2026-01-01   0.982177    0.982
   2 │ 2026-01-02  -0.692505   -0.693
   3 │ 2026-01-03  -0.811106   -0.811
```

The model row is timed at the training window's `stop`, 2025-01-01, and
[`readjls`](@ref) clips to the window it runs over, so over `later` alone every
prediction would be `missing`. `tolerance` widens the models' window back far
enough to reach the row, and also caps how old a model may be. A window starting
at the training `stop` needs no tolerance.

The same file gives the fit's diagnostics, over a window containing the model's
time:

```jldoctest model
fitted = Context(Date(2025, 1, 1), Date(2025, 1, 2))
readjls(path) |> modelreports() |> load(fitted)

# output

CausalFrame{Dates.Date} with 1 rows over [2025-01-01, 2025-01-02]
 Row │ time        report
     │ Date        Nothing
─────┼─────────────────────
   1 │ 2025-01-01
```
