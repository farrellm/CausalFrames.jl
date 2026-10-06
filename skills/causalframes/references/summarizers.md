# Summarizers

Contents: Row terms, `Count`, `CountDistinct`, `Sum`, `SumPower`, `Moment`, `Product`, `DotProduct`, `AgeWeightedSum`, `Mean`, `Variance`, `Std`, `Covariance`, `Correlation`, `LinearRegression`, `Quantile`, `Median`, `PercentRank`, `MeanAbsDev`, `SortedValues`, `WindowValues`, `Min`, `Max`, `MinIndex`, `MaxIndex`, `MinWithIndex`, `MaxWithIndex`, `First`, `Last`, `FitModel`, Summarizer interface.

Summarizers are what the [summarizing transforms](summarizing.md) fold over
rows. Each output column is named by suffix (`Sum(:mid)` produces `:mid_sum`)
and takes its element type from the input column.

## Row terms

A summarizer's input can be a per-row expression instead of a column: every
summarizer but `FitModel` takes a *row term* `name => f` wherever it takes a
column name (a `CausalFrames.ColumnSpec`). The term acts as a virtual
input column `name` holding `f(row)`, so its output is named as a column's
would be, and it widens, compensates and counts `missing` as a column does. The
transform computes each term once per row and never adds it to its output, so
there's no scratch column to drop afterwards.

Terms are identified by name, so every summarizer in one transform naming a
term must give it the same function (`===`): bind the function once, as in the
`CausalFrames.ColumnSpec` example, rather than writing the lambda
twice. A term may not share its name with an input column (including `:time`
and the key columns). A custom summarizer takes row terms by calling
`CausalFrames.withterms` in its constructor, and its
`CausalFrames.dependencies` may declare terms too.

## `Count`

    Count() -> Summarizer

The number of rows, in the column `:count` (`Int`, `0` for no rows). Alongside
`using MLJ`, which exports its own `Count`, write `CausalFrames.Count()`.

## `CountDistinct`

    CountDistinct(column::ColumnSpec) -> Summarizer

The number of distinct values of `column`, in `:{column}_countdistinct`
(`Int`, `0` for no rows).

`missing` counts as a value, so `1, missing, 1` has two distinct values; filter
out `missing` upstream for SQL's `count(DISTINCT x)`. The state holds every
distinct value seen, so memory is O(distinct values) per summary; in a sliding
window it also counts each value's rows, so it can drop a value when its last
row leaves.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `Sum`

    Sum(column::ColumnSpec) -> Summarizer

The sum of `column`, in `:{column}_sum` (`0` for no rows). The element type is
what `Base.sum` gives: small integers widen (`Int32` to `Int64`), other types
are kept (`Float32` stays `Float32`). Floats use compensated summation, with
NaN and ±Inf counted separately so a rolling window recovers once they leave.

#### Arguments
- `column`: a column name, or a row term `name => f` summing `f(row)` (see
  `ColumnSpec`).

```julia
df = DataFrame(time = 1:3, high = [5.0, 6.0, 8.0], low = [4.0, 4.5, 5.0])
p = readtable(df) |> summarize(Sum(:range => r -> r.high - r.low))
DataFrame(load(Context(0, 10), p))

# output

1×2 DataFrame
 Row │ time   range_sum
     │ Int64  Float64
─────┼──────────────────
   1 │    10        5.5
```

## `SumPower`

    SumPower(column::ColumnSpec, n::Integer) -> Summarizer

The sum of `column ^ n`, in `:{column}_sumpower_{n}` (`0` for no rows). The
element type follows `Sum` applied to `column ^ n`; each term is raised
in that widened type, so it cannot overflow the input type. `SumPower(:x, 1)`
is a separate column from `Sum(:x)`.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).
- `n`: the exponent, part of the output name.

## `Moment`

    Moment(column::ColumnSpec, n::Integer) -> Summarizer

The `n`-th raw moment of `column`, the mean of `column ^ n`, in
`:{column}_moment_{n}` (`missing` for no rows). Computed from
`SumPower``(column, n)` and `Count`; integer input gives
`Float64`.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).
- `n`: the order, part of the output name.

## `Product`

    Product(column::ColumnSpec) -> Summarizer

The product of `column`, in `:{column}_product` (`1` for no rows). The element
type is what `Base.prod` gives: small integers widen (`Int32` to `Int64`),
other types are kept.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `DotProduct`

    DotProduct(a::ColumnSpec, b::ColumnSpec) -> Summarizer

The sum of `a * b`, in `:{a}_{b}_dotproduct` (`0` for no rows). The element type
widens as for `Sum`, and terms are formed in that type. `DotProduct(:y,
:x)` shares the accumulator of `DotProduct(:x, :y)` (and of any
`Covariance` or `LinearRegression` needing it).

#### Arguments
- `a`, `b`: column names or row terms `name => f` (see `ColumnSpec`).
  A row term is named, and so ordered, by its `name`.

```julia
df = DataFrame(time = 1:2, high = [6.0, 9.0], low = [4.0, 6.0],
    volume = [10, 20])
tp = r -> (r.high + r.low) / 2
p = readtable(df) |> summarize(DotProduct(:tp => tp, :volume))
DataFrame(load(Context(0, 10), p))

# output

1×2 DataFrame
 Row │ time   tp_volume_dotproduct
     │ Int64  Float64
─────┼─────────────────────────────
   1 │    10                 200.0
```

## `AgeWeightedSum`

    AgeWeightedSum(column::ColumnSpec) -> Summarizer

The sum of `column` weighted by each row's age, `Σₖ k·yₖ` where `k` counts the
rows folded after row `k` (`0` for the newest), in `:{column}_ageweightedsum`
(`0` for no rows). With `Sum` and `Count` it gives the linearly
weighted moving average `(n·Σy − Σk·y) / (n(n+1)/2)`, whose newest row weighs
`n`, and a least-squares fit on the row number.

The element type follows `Sum`, and floats use compensated summation
with NaN and ±Inf counted separately, so a rolling window recovers once they
leave. The newest row weighs `0`, so a NaN or ±Inf there contributes nothing
until a later row arrives; a `missing` anywhere gives `missing`.

#### Arguments
- `column`: the column to weight, or a row term `name => f` (see `ColumnSpec`).
  Age is counted in rows, in stream order, so
  rows tied in time have different ages.

```julia
df = DataFrame(time = 1:4, y = [1.0, 2.0, 3.0, 4.0])
p = readtable(df) |> addsummarycolumns(AgeWeightedSum(:y))
DataFrame(load(Context(0, 10), p))

# output

4×3 DataFrame
 Row │ time   y        y_ageweightedsum
     │ Int64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1      1.0               0.0
   2 │     2      2.0               1.0
   3 │     3      3.0               4.0
   4 │     4      4.0              10.0
```

## `Mean`

    Mean(column::ColumnSpec) -> Summarizer

The mean of `column`, in `:{column}_mean` (`missing` for no rows). Computed from
`Sum``(column)` and `Count`; integer input gives `Float64`,
`Float32` stays `Float32`.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `Variance`

    Variance(column::ColumnSpec; corrected = true) -> Summarizer

The variance of `column`, as `Statistics.var`, in `:{column}_variance`
(`missing` for no rows, `NaN` if a value is NaN or ±Inf). Integer input gives
`Float64`. The state folds deviations from a shift near the data, so a large
level beside a small spread doesn't cancel, and round-off never makes the
variance negative.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Keywords
- `corrected = true`: divide by `n - 1` (a single row gives `NaN`); `false`
  divides by `n`. It is not part of the output name, so both forms of one column
  cannot be requested together.

## `Std`

    Std(column::ColumnSpec; corrected = true) -> Summarizer

The standard deviation of `column`, as `Statistics.std`, in `:{column}_std`
(`missing` for no rows): the square root of `Variance`, with round-off
negatives clamped to zero.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Keywords
- `corrected = true`: as for `Variance`.

## `Covariance`

    Covariance(a::ColumnSpec, b::ColumnSpec; corrected = true) -> Summarizer

The covariance of `a` and `b`, as `Statistics.cov`, in `:{a}_{b}_covariance`
(`missing` for no rows, `NaN` if a value is NaN or ±Inf). Like
`Variance`, it folds deviations from a shift near the data, so large
levels don't cancel.

#### Arguments
- `a`, `b`: column names or row terms `name => f` (see `ColumnSpec`).

#### Keywords
- `corrected = true`: as for `Variance`, including that it is not part
  of the output name.

## `Correlation`

    Correlation(a::ColumnSpec, b::ColumnSpec) -> Summarizer

The Pearson correlation of `a` and `b`, as `Statistics.cor`, clamped to
`[-1, 1]`, in `:{a}_{b}_correlation` (`missing` for no rows, `NaN` for one).
Computed from `Covariance` and the two columns' `Std`s; there is
no `corrected` keyword, since the correction cancels.

#### Arguments
- `a`, `b`: column names or row terms `name => f` (see `ColumnSpec`).

## `LinearRegression`

    LinearRegression(predictors, response::ColumnSpec; intercept = true,
                     name = nothing) -> Summarizer

An ordinary least squares fit of `response` on `predictors`.

#### Arguments
- `predictors`: a column name or row term `name => f` (see `ColumnSpec`),
  or a collection of distinct ones (`K` of them). None may be named
  `intercept`.
- `response`: the response column, or a row term.

#### Keywords
- `intercept = true`: fit a constant term. Without it, `r2` is the uncentered
  R².
- `name = nothing`: a prefix for every output column, `{name}_{column}`.
  Needed to request two regressions together, since both would otherwise
  produce `n`, `r2` and `stderr`.

#### Output columns

| column | meaning |
|---|---|
| `n` | rows folded (`Int`, never `missing`) |
| `r2` | coefficient of determination |
| `stderr` | residual standard error |
| `intercept_beta`, `intercept_tstat` | the constant term and its t statistic (only with `intercept`) |
| `{p}_beta`, `{p}_tstat` | per predictor `p`, in order |

```julia
df = DataFrame(time = 1:5, x = [1.0, 2.0, 3.0, 4.0, 5.0], z = [0.0, 1.0, 0.0, 1.0, 1.0],
               y = [3.1, 7.9, 7.2, 12.1, 13.8])
p = readtable(df) |> summarize(LinearRegression([:x, :z], :y; name = :m1))
names(load(Context(0, 10), p))

# output

10-element Vector{String}:
 "time"
 "m1_n"
 "m1_r2"
 "m1_stderr"
 "m1_intercept_beta"
 "m1_intercept_tstat"
 "m1_x_beta"
 "m1_x_tstat"
 "m1_z_beta"
 "m1_z_tstat"
```

The statistics share one element type (`Float64` for integer input) and admit
`missing` if an input does. No rows, or a `missing` in any input, gives
`missing` statistics. A rank-deficient fit (collinear predictors, or too few
rows) gives `NaN` rather than an error; with no residual degrees of freedom the
coefficients are exact but `stderr` and the t statistics are `NaN`.

With an intercept the fit is computed from the same centred accumulators as
`Variance` and `Covariance`, so large levels don't cancel, plus
`Count` and each column's `Sum` for the means. Without one it
is computed from `Count`, `SumPower``(·, 2)` and
`DotProduct`. Either way, regressions over overlapping columns, and the
statistics beside them, share accumulators. With `K ≥ 2` each emitted row allocates a `K × K`
workspace; `K = 1` allocates nothing.

## `Quantile`

    Quantile(column::ColumnSpec, p; interpolation = :linear) -> Summarizer

The `p` quantiles of `column`, one column per probability, in
`:{column}_quantile_{100p}` (`:x_quantile_50` for `0.5`, `:x_quantile_2_5` for
`0.025`; `missing` for no rows). A `missing` in the rows gives `missing` and a
NaN gives NaN; in a sliding window the quantiles recover once that row leaves.
Computed from `CausalFrames.SortedValues`, so memory is
O(window) per key.

#### Arguments
- `column`: the column to summarize, or a row term `name => f` (see `ColumnSpec`).
  Its values must be ordered by `isless`,
  and `:linear` must be able to interpolate them.
- `p`: a probability in `[0, 1]`, or a non-empty collection of distinct ones,
  giving one column each in the order given. Any other value is an
  `ArgumentError`.

#### Keywords
- `interpolation = :linear`: how to pick the quantile from the sorted values.
  `:linear` interpolates between them as `Statistics.quantile` does by default
  (Hyndman and Fan's type 7, up to rounding), so integer input gives `Float64`. `:nearestrank`
  takes the `k`-th smallest value, for the smallest `k` with `k/n ≥ p` (type 1,
  and TA-Lib's `PERCENTILE`), keeping `column`'s element type. It compares `k/n`
  and `p` in floating point, so `p = 0.07` over 100 rows gives the 7th value.
  Any other symbol is an `ArgumentError`. It is not part of the output name, so
  both forms of one `p` cannot be requested together.

```julia
df = DataFrame(time = 1:5, x = [4, 1, 3, 5, 2])
p = readtable(df) |>
    addrollingcolumns((w3 = 3,), Quantile(:x, [0.25, 0.5]))
DataFrame(load(Context(0, 10), p))

# output

5×4 DataFrame
 Row │ time   x      w3_x_quantile_25  w3_x_quantile_50
     │ Int64  Int64  Float64?          Float64?
─────┼──────────────────────────────────────────────────
   1 │     1      4              4.0                4.0
   2 │     2      1              1.75               2.5
   3 │     3      3              2.0                3.0
   4 │     4      5              2.5                3.5
   5 │     5      2              1.75               2.5
```

## `Median`

    Median(column::ColumnSpec) -> Summarizer

The median of `column`, as `Statistics.median` up to rounding, in `:{column}_median`
(`missing` for no rows): `Quantile``(column, 0.5)` under its own name,
sharing its accumulator, so integer input gives `Float64`. A `missing` in the
rows gives `missing` and a NaN gives NaN.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

```julia
df = DataFrame(time = 1:4, x = [4, 1, 3, 5])
DataFrame(load(Context(0, 10), readtable(df) |> summarize(Median(:x))))

# output

1×2 DataFrame
 Row │ time   x_median
     │ Int64  Float64
─────┼─────────────────
   1 │    10       3.5
```

## `PercentRank`

    PercentRank(column::ColumnSpec) -> Summarizer

The fraction of the other rows whose `column` is strictly below the newest
row's, in `:{column}_percentrank` (`Float64`, `missing` for no rows): for `n`
rows, `count(< newest) / (n - 1)`, so `0` for a strict minimum, `1` for a strict
maximum and `NaN` for a single row. This is Excel's `PERCENTRANK.INC` of the
newest value, and TA-Lib's `PERCENTRANK` over a period of `n - 1` divided by 100.
A `missing` in the rows gives `missing` and a NaN gives NaN; in a sliding window
it recovers once that row leaves. Computed from
`CausalFrames.SortedValues` and `Last`, so memory
is O(window) per key.

#### Arguments
- `column`: the column to rank, or a row term `name => f` (see `ColumnSpec`).
  Its values must be ordered by `isless`.

```julia
df = DataFrame(time = 1:5, x = [4, 1, 3, 5, 3])
p = readtable(df) |> addrollingcolumns((w4 = 4,), PercentRank(:x))
DataFrame(load(Context(0, 10), p))

# output

5×3 DataFrame
 Row │ time   x      w4_x_percentrank
     │ Int64  Int64  Float64?
─────┼────────────────────────────────
   1 │     1      4            NaN
   2 │     2      1              0.0
   3 │     3      3              0.5
   4 │     4      5              1.0
   5 │     5      3              0.25
```

## `MeanAbsDev`

    MeanAbsDev(column::ColumnSpec) -> Summarizer

The mean absolute deviation of `column` about its mean, `Σ|x - mean| / n`, in
`:{column}_meanabsdev` (`missing` for no rows). This is Excel's `AVEDEV`,
TA-Lib's `AVGDEV` and the deviation in its `CCI`. Integer input gives `Float64`,
and `Float32` stays `Float32`. A `missing` in the rows gives `missing` and a NaN
gives NaN, and in a sliding window it recovers once that row leaves. It is
computed from `CausalFrames.WindowValues` and
`Mean`, so memory is O(window) per key, and each emission scans the
window's values once.

#### Arguments
- `column`: the column to summarize, or a row term `name => f` (see `ColumnSpec`).

```julia
df = DataFrame(time = 1:5, x = [1, 2, 6, 3, 3])
p = readtable(df) |> addrollingcolumns((b3 = Bars(3),), MeanAbsDev(:x))
DataFrame(load(Context(0, 10), p))

# output

5×3 DataFrame
 Row │ time   x      b3_x_meanabsdev
     │ Int64  Int64  Float64?
─────┼───────────────────────────────
   1 │     1      1    missing
   2 │     2      2    missing
   3 │     3      6          2.0
   4 │     4      3          1.55556
   5 │     5      3          1.33333
```

## `SortedValues`

    SortedValues(column::ColumnSpec) -> Summarizer

The values of `column` in sorted order: the accumulator `Quantile`,
`Median` and `PercentRank` read, for a summarizer of your own to
depend on too. It is not exported; write `CausalFrames.SortedValues`.

Its value, in `:{column}_sortedvalues` (`missing` for no rows), is its state
itself, *borrowed*: read it within the same emission and never keep it. Don't
request it as an output column, as every row would share the one state. The
state has three fields to read:

- `vals`: the values that are neither `missing` nor NaN, sorted by `isless`, as
  a `Vector` of `column`'s non-missing element type;
- `missings`, `nans`: how many `missing` and NaN values are folded in.

Memory is O(rows) per summary, so O(window) in a sliding window, where adding
and removing a row each cost a binary search and a shift of up to the window's
values.

#### Arguments
- `column`: the column to collect, or a row term `name => f` (see `ColumnSpec`).
  Its values must be ordered by `isless`.

```julia
st = CausalFrames.fresh(CausalFrames.SortedValues(:x), (; x = Float64))
foreach(v -> CausalFrames.update!(st, (; x = v)), [3.0, NaN, 1.0, 2.0])
sv = CausalFrames.value(st).x_sortedvalues
(sv.vals, sv.nans)

# output

([1.0, 2.0, 3.0], 1)
```

## `WindowValues`

    WindowValues(column::ColumnSpec) -> Summarizer

The values of `column` in the order they were folded: the accumulator
`MeanAbsDev` reads, for a summarizer of your own to depend on too when
it needs every value but not their order. It is not exported; write
`CausalFrames.WindowValues`.

Its value, in `:{column}_windowvalues` (`missing` for no rows), is its state
itself, *borrowed*: read it within the same emission and never keep it. Don't
request it as an output column, as every row would share the one state. The
state has these fields to read:

- `vals` and `head`: the values that are neither `missing` nor NaN are
  `vals[head:end]`, oldest first, in a `Vector` of `column`'s non-missing
  element type;
- `missings`, `nans`: how many `missing` and NaN values are folded in.

Memory is O(rows) per summary, so O(window) in a sliding window. Adding a row
appends and removing one advances `head`, each O(1) amortized: removal relies
on `downdate!`'s oldest-first law. Where the order is needed, use
`CausalFrames.SortedValues` instead.

#### Arguments
- `column`: the column to collect, or a row term `name => f` (see `ColumnSpec`).

```julia
st = CausalFrames.fresh(CausalFrames.WindowValues(:x), (; x = Float64))
foreach(v -> CausalFrames.update!(st, (; x = v)), [3.0, NaN, 1.0, 2.0])
wv = CausalFrames.value(st).x_windowvalues
(wv.vals[wv.head:end], wv.nans)

# output

([3.0, 1.0, 2.0], 1)
```

## `Min`

    Min(column::ColumnSpec) -> Summarizer

The minimum of `column`, in `:{column}_min`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the minimum, up to the whole window when `column` rises.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `Max`

    Max(column::ColumnSpec) -> Summarizer

The maximum of `column`, in `:{column}_max`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the maximum, up to the whole window when `column` falls.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `MinIndex`

    MinIndex(column::ColumnSpec) -> Summarizer

How many rows ago `column` took its minimum, in `:{column}_minindex`, as an
`Int` (`missing` for no rows): `0` is the newest row. Ties go to the most recent
row. A `NaN` or `missing` is the minimum, as for `Min`, so its position
is reported while it is folded. To get the minimum as well, use
`MinWithIndex`, which keeps one state for both.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Examples
```julia
df = DataFrame(time = 1:6, x = [3, 1, 4, 1, 5, 9])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MinIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×3 DataFrame
 Row │ time   x      w3_x_minindex
     │ Int64  Int64  Int64?
─────┼─────────────────────────────
   1 │     1      3              0
   2 │     2      1              0
   3 │     3      4              1
   4 │     4      1              0
   5 │     5      5              1
   6 │     6      9              2
```

## `MaxIndex`

    MaxIndex(column::ColumnSpec) -> Summarizer

How many rows ago `column` took its maximum, in `:{column}_maxindex`, as an
`Int` (`missing` for no rows): `0` is the newest row. Ties go to the most recent
row. A `NaN` or `missing` is the maximum, as for `Max`, so its position
is reported while it is folded. To get the maximum as well, use
`MaxWithIndex`, which keeps one state for both.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Examples
```julia
df = DataFrame(time = 1:6, x = [3, 5, 2, 5, 1, 4])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MaxIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×3 DataFrame
 Row │ time   x      w3_x_maxindex
     │ Int64  Int64  Int64?
─────┼─────────────────────────────
   1 │     1      3              0
   2 │     2      5              0
   3 │     3      2              1
   4 │     4      5              0
   5 │     5      1              1
   6 │     6      4              2
```

## `MinWithIndex`

    MinWithIndex(column::ColumnSpec) -> Summarizer

The minimum of `column` and how many rows ago it was taken, in `:{column}_min`
(with `column`'s element type) and `:{column}_minindex` (an `Int`), both
`missing` for no rows. It folds one state for both, where `Min` and
`MinIndex` together fold two, and so can't be requested with `Min` of
the same column. The index follows `MinIndex`.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Examples
```julia
df = DataFrame(time = 1:6, x = [3, 1, 4, 1, 5, 9])
p = readtable(df) |> summarize(MinWithIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

1×3 DataFrame
 Row │ time   x_min  x_minindex
     │ Int64  Int64  Int64
─────┼──────────────────────────
   1 │    10      1           2
```

## `MaxWithIndex`

    MaxWithIndex(column::ColumnSpec) -> Summarizer

The maximum of `column` and how many rows ago it was taken, in `:{column}_max`
(with `column`'s element type) and `:{column}_maxindex` (an `Int`), both
`missing` for no rows. It folds one state for both, where `Max` and
`MaxIndex` together fold two, and so can't be requested with `Max` of
the same column. The index follows `MaxIndex`.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

#### Examples
```julia
df = DataFrame(time = 1:6, x = [3, 5, 2, 5, 1, 4])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MaxWithIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×4 DataFrame
 Row │ time   x      w3_x_max  w3_x_maxindex
     │ Int64  Int64  Int64?    Int64?
─────┼───────────────────────────────────────
   1 │     1      3         3              0
   2 │     2      5         5              0
   3 │     3      2         5              1
   4 │     4      5         5              0
   5 │     5      1         5              1
   6 │     6      4         5              2
```

## `First`

    First(column::ColumnSpec) -> Summarizer

The value of `column` in the first row folded, in `:{column}_first`, with
`column`'s element type (`missing` for no rows). In a sliding window it keeps
every value in the window, O(window) memory per key.

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `Last`

    Last(column::ColumnSpec) -> Summarizer

The value of `column` in the last row folded, in `:{column}_last`, with
`column`'s element type (`missing` for no rows).

#### Arguments
- `column`: a column name, or a row term `name => f` (see `ColumnSpec`).

## `FitModel`

    FitModel(model, predictors, response::Symbol; name = :model,
             verbosity = 0) -> Summarizer

Fits an MLJ model to the rows it summarizes, emitting a `FittedModel`
(`missing` for no rows). Requires MLJModelInterface: `using MLJ`, or an MLJ
model package (most models also need MLJBase, which `using MLJ` loads).

#### Arguments
- `model`: an `MLJModelInterface.Model`; anything else is an `ArgumentError`.
- `predictors`: a column name, or a non-empty collection of distinct names.
  They reach the model as a column table of their own element types, without
  scientific-type coercion (coerce upstream with `addcolumns`), and
  `missing` is passed through. A row term (see `ColumnSpec`) is an
  `ArgumentError`: `applymodels` predicts from another stream, which
  would lack its column.
- `response`: the response column, which may not be a predictor.

#### Keywords
- `name = :model`: the output column.
- `verbosity = 0`: passed to the model's `fit`.

Rows are buffered and the model is fit each time a summary is emitted: once
under `summarize`, per interval under `intervalize`, per tick
and key under `summarizewindows`, and after every row (quadratic)
under `addsummarycolumns`. Windows re-fold for it, as it neither
combines nor inverts. Apply the models with `applymodels`, or fit and
apply in one step with `addpredictions`.

## Summarizer interface

To define a summarizer, subtype `Summarizer` and extend these functions
(unexported: `CausalFrames.fresh` and so on).

    Summarizer

Abstract supertype of summarizers: immutable configurations (typically a column
name, held as a type parameter so the output names are known to the compiler)
that the summarizing transforms fold over rows. A summarizer implements:

- `emptyvalue``(s)`: the summary of no rows, a `NamedTuple` keyed by
  output column name;
- `fresh``(s, intypes)`: a zero `SummarizerState` for input
  columns with the element types `intypes`;
- optionally `dependencies``(s)`, with the two-argument
  `value`, to compute its value from other summarizers'.

Summarizers with the same output names are treated as the same summarizer and
share state. Dependencies are folded but appear in the output only if
requested. Subtype `MonoidSummarizer` or `GroupSummarizer` to
unlock faster window algorithms.

---

    MonoidSummarizer <: Summarizer

A summarizer whose states `combine!` associatively, with a
`fresh` state as identity. Window transforms answer each window from a
segment tree of partial states — O(log window) — instead of re-folding it.

---

    GroupSummarizer <: MonoidSummarizer

A monoid summarizer whose windowed state (`freshwindowed`) can remove
its oldest row with `downdate!`. Window transforms slide such a state
in O(1) amortized per row, removing rows as they leave the window, unless
`isinvertible` says the state can no longer be inverted.

---

    SummarizerState

Abstract supertype of the running state of one summarization, built by
`fresh``(s, intypes)`. Its value fields are concretely typed, so the
output columns are too. A state implements:

- `fresh``(st)`: a zero state of the same type;
- `update!``(st, row)`: fold in one row;
- `value``(st)`: the current summary, a `NamedTuple` keyed by output
  column name;
- optionally `fresh!``(st)`, `widenstate``(st, intypes)`, and,
  for structured summarizers, `combine!`, `downdate!` and
  `isinvertible`.

A `GroupSummarizer` may also build a separate windowed state with
`freshwindowed`.

---

    emptyvalue(s::Summarizer) -> NamedTuple

The summary of no rows, keyed by output column name. Transforms read output
names from it before any data arrives, and use it when there is no data to
build a state from.

---

    fresh(s::Summarizer, intypes::NamedTuple) -> SummarizerState
    fresh(st::SummarizerState) -> SummarizerState

A zero state. The first form builds it from `s` and the input column element
types `intypes`, read as `update!` reads the row (`intypes[column]` for
`row[column]`). The second returns a new zero state of the same concrete type as
`st`, used for each key group and cycle. Transforms never mutate the
summarizers they are given.

---

    fresh!(st::SummarizerState) -> SummarizerState

Zero `st` in place, where possible, and return it; callers use the returned
value. The default returns `fresh(st)`. Implementing it avoids an allocation per
state per cycle, window query or row in the window transforms. A state whose
value is only read after a row has been folded (`Min`, `First`, …) may keep a
stale value.

---

    freshwindowed(s::Summarizer, intypes::NamedTuple) -> SummarizerState

A zero state for a sliding window, which the window transforms
`update!` as rows arrive and `downdate!` as they leave, oldest
first. Defaults to `fresh``(s, intypes)`. A
`GroupSummarizer` whose ordinary state cannot remove rows implements it
instead: the windowed `First` keeps every value in the window, where the
ordinary one keeps a single value.

The windowed state needs `fresh`, `fresh!`, `update!`, `downdate!`, `value` and
optionally `isinvertible`, and its `value` must have the ordinary state's type.
It is never combined or widened.

---

    update!(st::SummarizerState, row)

Fold one row into `st`. `row` supports `row.name` and `row[:name]`, including
`row.time`.

---

    value(st::SummarizerState) -> NamedTuple
    value(st::SummarizerState, vals::NamedTuple) -> NamedTuple

The current summary, keyed by output column name; its value types are the
output column element types. Only called on a state that has folded at least
one row (the summary of no rows is `emptyvalue`).

The two-argument form also receives `vals`, the values of every summarizer
earlier in dependency order, including all of `dependencies``(s)`. It
defaults to the one-argument form; a dependent summarizer implements it instead.

---

    widenstate(st::SummarizerState, intypes::NamedTuple) -> SummarizerState

`st` rebuilt for the wider input element types `intypes`, keeping its
accumulated value. Called when a later chunk widens a column's type (say `Int`
to `Float64`). Defaults to returning `st`, which suits states whose type does
not depend on the input.

---

    dependencies(s::Summarizer) -> Tuple

The summarizers whose values `s` reads in the two-argument `value`.
They are expanded recursively and deduplicated by output name, so a dependency
also requested by the user is folded once, and they appear in the output only
if requested. Defaults to `()`.

---

    combine!(dest::SummarizerState, a::SummarizerState, b::SummarizerState)

Set `dest` to the state folding `a`'s rows then `b`'s would give. Required for
a `MonoidSummarizer`: it must be associative, with a `fresh`
state as identity on both sides. Every row of `a` precedes every row of `b` in
stream order, so order-sensitive states (`First`, `Last`) can combine. All three
states have the same type and `dest` may alias `a` or `b`, so read the inputs
before writing.

---

    downdate!(st::SummarizerState, row)

Remove `row`, the oldest row still folded into `st`, inverting its
`update!`. Callers remove rows in the order they folded them, so a state
may rely on that: `AgeWeightedSum` and the windowed `Min`, `First`
and `Last` do. Required of a `GroupSummarizer`'s windowed state
(`freshwindowed`). The built-in sums are exact for integers; for floats
they use compensated summation and count NaN, ±Inf and `missing` terms
separately, so those remove exactly and finite terms leave only small round-off.

---

    isinvertible(st::SummarizerState) -> Bool

Whether `downdate!` inverts `update!` for this windowed
state's actual accumulator type. Defaults to `true`; a state that folds an
absorbing value it cannot recover from returns `false`, and window transforms
then fold that summarizer through a segment tree instead.

---

    barwindow(s::Summarizer, n::Integer, intypes::NamedTuple) -> SummarizerState

A state summarizing only the last `n` rows folded into it, for a summarizer
whose own state needs a trailing window (a moving average inside a recursive
indicator, say). `update!` folds a row in and drops the row `n` rows
older; `value` is `s`'s value over the last `n` rows, with every
column `missing` until `n` rows have arrived, so each value column is
`Union{Missing, T}`.

`s`'s dependencies are expanded and shared as in the summarizing transforms.
Groups slide in O(1) per row and other monoids take O(1) amortized, and neither
allocates. The state's type depends on `s`, `n`'s tier choice and `intypes`,
not on their types alone, so an outer state holding one should take its type
as a type parameter.

#### Arguments
- `s`: a `MonoidSummarizer`, which may read row terms; any other
  summarizer, or a dependency that is one, is an `ArgumentError`.
- `n`: the window's row count, at least 1.
- `intypes`: the element types of the columns the rows passed to `update!`
  carry, as in `fresh`. A row term's type is inferred from them and
  must be concrete, or a `Union` of concrete types; its name may not be one
  of theirs.

```julia
st = CausalFrames.barwindow(Mean(:x), 2, (x = Float64,))
map(1.0:4.0) do x
    CausalFrames.update!(st, (; x))
    CausalFrames.value(st).x_mean
end

# output

4-element Vector{Union{Missing, Float64}}:
  missing
 1.5
 2.5
 3.5
```

Rows leave the window oldest first, so a wrapped `downdate!` sees the
order it expects. `widenstate` re-types the state and keeps its window.

---

    ColumnSpec

An input column as a summarizer constructor takes it: a column name (a
`Symbol`), or a *row term* `name => f`. A row term is a virtual input column
`name`, whose value in each row is `f(row)`. The summarizer reads it as it
would a column, so its output is named after `name` (`Sum(:range => f)` gives
`:range_sum`). The summarizing transforms compute each row term once per row,
alongside the input, and never add it to their output. See
Row terms.

`Union{Symbol, Pair{Symbol, <:Base.Callable}}`.

```julia
df = DataFrame(time = 1:3, high = [5.0, 6.0, 8.0], low = [4.0, 4.5, 5.0])
spread = r -> r.high - r.low   # bound once, so both summarizers share it
p = readtable(df) |>
    addsummarycolumns([Sum(:spread => spread), Mean(:spread => spread)])
DataFrame(load(Context(0, 10), p))

# output

3×5 DataFrame
 Row │ time   high     low      spread_sum  spread_mean
     │ Int64  Float64  Float64  Float64     Float64
─────┼──────────────────────────────────────────────────
   1 │     1      5.0      4.0         1.0      1.0
   2 │     2      6.0      4.5         2.5      1.25
   3 │     3      8.0      5.0         5.5      1.83333
```

---

    withterms(s::Summarizer, specs::ColumnSpec...) -> Summarizer

`s`, carrying the row terms among `specs`, which are the `ColumnSpec`s
`s` was built from. A constructor taking a column calls it on the summarizer
built from the specs' names:

```julia
MySum(column::ColumnSpec) = withterms(MySum{colname(column)}(), column)
```

With no row term among `specs` it returns `s` itself. A row term may not be
named `:time`, and two row terms with the same name must be the same function;
either is an `ArgumentError` naming `s`'s type.
