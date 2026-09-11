# Summarizers

The values the [summarizing transforms](summarizing.md) fold over their rows.
An output column takes its element type from the input column, and is named by
suffix — `Sum(:mid)` produces `:mid_sum`.

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

Extend these (unexported — `CausalFrames.fresh` etc.) to define a custom
summarizer.

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
