# Summarizers

Summarizers are what the [summarizing transforms](summarizing.md) fold over
rows. Each output column is named by suffix (`Sum(:mid)` produces `:mid_sum`)
and takes its element type from the input column.

## Row terms

A summarizer's input can be a per-row expression instead of a column: every
summarizer but `FitModel` takes a *row term* `name => f` wherever it takes a
column name (a [`CausalFrames.ColumnSpec`](@ref)). The term acts as a virtual
input column `name` holding `f(row)`, where `row` is a `NamedTuple` of the
input row's columns, so its output is named as a column's
would be, and it widens, compensates and counts `missing` as a column does. Its
element type is inferred from `f` (`Union{Float64, Int64}` for
`r -> r.x > 2 ? r.x : 1` over a `Float64` column), or taken from its values when
inference gives up. The
transform computes each term once per row and never adds it to its output, so
there's no scratch column to drop afterwards.

Terms are identified by name, so every summarizer in one transform naming a
term must give it the same function (`===`): bind the function once, as in the
[`CausalFrames.ColumnSpec`](@ref) example, rather than writing the lambda
twice. A term may not share its name with an input column (including `:time`
and the key columns). A custom summarizer takes row terms by calling
[`CausalFrames.withterms`](@ref) in its constructor, and its
[`CausalFrames.dependencies`](@ref) may declare terms too.

A term's `row` holds every input column. A term the compiler inlines pays only
for the fields it reads. A term it doesn't inline (a large, recursive or
`@noinline` function) pays for every column on every row. On a table with
dozens of columns, that can cost several times the term itself. On a wide
table, do one of these:

- mark the term function `@inline` where it is defined;
- keep the term small, reading its fields and passing them on, as in
  `r -> f(r.high, r.low)`;
- drop the columns it doesn't need with [`selectcolumns`](@ref) upstream, when
  the transform's output doesn't keep the input columns ([`summarize`](@ref),
  [`summarizecycles`](@ref), [`intervalize`](@ref),
  [`summarizewindows`](@ref));
- compute the value as a column with [`addcolumns`](@ref) and summarize that
  column.

## `Count`

```@docs
Count
```

## `CountDistinct`

```@docs
CountDistinct
```

## `Sum`

```@docs
Sum
```

## `SumPower`

```@docs
SumPower
```

## `Moment`

```@docs
Moment
```

## `Product`

```@docs
Product
```

## `DotProduct`

```@docs
DotProduct
```

## `AgeWeightedSum`

```@docs
AgeWeightedSum
```

## `Mean`

```@docs
Mean
```

## `Variance`

```@docs
Variance
```

## `Std`

```@docs
Std
```

## `Covariance`

```@docs
Covariance
```

## `Correlation`

```@docs
Correlation
```

## `LinearRegression`

```@docs
LinearRegression
```

## `Quantile`

```@docs
Quantile
```

## `Median`

```@docs
Median
```

## `PercentRank`

```@docs
PercentRank
```

## `MeanAbsDev`

```@docs
MeanAbsDev
```

## `SortedValues`

```@docs
CausalFrames.SortedValues
```

## `WindowValues`

```@docs
CausalFrames.WindowValues
```

## `Min`

```@docs
Min
```

## `Max`

```@docs
Max
```

## `MinIndex`

```@docs
MinIndex
```

## `MaxIndex`

```@docs
MaxIndex
```

## `MinWithIndex`

```@docs
MinWithIndex
```

## `MaxWithIndex`

```@docs
MaxWithIndex
```

## `First`

```@docs
First
```

## `Last`

```@docs
Last
```

## `FitModel`

```@docs
FitModel
```

## Summarizer interface

To define a summarizer, subtype `Summarizer` and extend these functions
(unexported: `CausalFrames.fresh` and so on).

```@docs
Summarizer
MonoidSummarizer
GroupSummarizer
SummarizerState
CausalFrames.emptyvalue
CausalFrames.fresh
CausalFrames.fresh!
CausalFrames.freshwindowed
CausalFrames.update!
CausalFrames.value
CausalFrames.widenstate
CausalFrames.dependencies
CausalFrames.inputcolumns
CausalFrames.combine!
CausalFrames.downdate!
CausalFrames.isinvertible
CausalFrames.barwindow
CausalFrames.barinputcolumns
CausalFrames.barterms
CausalFrames.rowterms
CausalFrames.ColumnSpec
CausalFrames.withterms
```
