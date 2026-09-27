# Summarizers

Summarizers are what the [summarizing transforms](summarizing.md) fold over
rows. Each output column is named by suffix (`Sum(:mid)` produces `:mid_sum`)
and takes its element type from the input column.

## Row terms

A summarizer's input can be a per-row expression instead of a column: every
summarizer but `FitModel` takes a *row term* `name => f` wherever it takes a
column name (a [`CausalFrames.ColumnSpec`](@ref)). The term acts as a virtual
input column `name` holding `f(row)`, so its output is named as a column's
would be, and it widens, compensates and counts `missing` as a column does. The
transform computes each term once per row and never adds it to its output, so
there's no scratch column to drop afterwards.

Terms are identified by name, so every summarizer in one transform naming a
term must give it the same function (`===`): bind the function once, as in the
[`CausalFrames.ColumnSpec`](@ref) example, rather than writing the lambda
twice. A term may not share its name with an input column (including `:time`
and the key columns). A custom summarizer takes row terms by calling
[`CausalFrames.withterms`](@ref) in its constructor, and its
[`CausalFrames.dependencies`](@ref) may declare terms too.

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

## `SortedValues`

```@docs
CausalFrames.SortedValues
```

## `Min`

```@docs
Min
```

## `Max`

```@docs
Max
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
CausalFrames.combine!
CausalFrames.downdate!
CausalFrames.isinvertible
CausalFrames.barwindow
CausalFrames.ColumnSpec
CausalFrames.withterms
```
