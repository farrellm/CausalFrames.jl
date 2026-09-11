# Model fitting (MLJ)

With an MLJ model package loaded (`using MLJ`), [`FitModel`](@ref "FitModel")
fits a model inside any summarization transform, emitting
[`FittedModel`](@ref)s; these operators apply such models to a stream and read
their diagnostics.

## `applymodels`

```@docs
applymodels
```

## `addpredictions`

```@docs
addpredictions
```

## `modelreports`

```@docs
modelreports
```

## Fitted models

```@docs
FittedModel
```
