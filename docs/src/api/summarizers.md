# Summarizers

Summarizers are what the [summarizing transforms](summarizing.md) fold over
rows. Each output column is named by suffix (`Sum(:mid)` produces `:mid_sum`)
and takes its element type from the input column.

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
CausalFrames.update!
CausalFrames.value
CausalFrames.widenstate
CausalFrames.dependencies
CausalFrames.combine!
CausalFrames.downdate!
CausalFrames.isinvertible
```
