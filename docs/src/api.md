# API reference

## Core types

```@docs
Context
timetype
CausalFrame
context
DataFrames.DataFrame(::CausalFrame)
```

## Pipelines

```@docs
CausalPipeline
load
stream
scan
```

## Sources

```@docs
emptyframe
concatenate
Base.merge(::CausalPipeline, ::CausalPipeline...)
clock
readcsv
readparquet
readjls
```

## Sinks

```@docs
writecsv
writeparquet
writejls
```

## Row-wise transforms

```@docs
filterrows
addcolumns
selectcolumns
dropcolumns
reordercolumns
```

## Filling

Two ways to resolve `missing` values — the ones a [`merge`](@ref) schema
union, an [`asofjoin`](@ref) non-match or a nullable parquet column leaves
behind. Only `fillmissing` is row-wise; `forwardfill` carries the last
non-missing value of every filled column across rows, chunks and keys.

```@docs
forwardfill
fillmissing
```

## Truncation and reduction

Neither of these is row-wise: both carry state across the whole window, `head`
a remaining-row budget and `lastrow` a store of rows per key.

```@docs
head
lastrow
```

## Joins

```@docs
asofjoin
```

The forward-looking join lives in the `CausalFrames.Acausal` submodule and is
not re-exported — reach it with `using CausalFrames.Acausal`.

```@docs
CausalFrames.Acausal.futurejoin
```

## Lead and lag

```@docs
lag
```

`lead` shifts forward in time and so is acausal; like `futurejoin` it lives in
the `CausalFrames.Acausal` submodule and is not re-exported — reach it with
`using CausalFrames.Acausal`.

```@docs
CausalFrames.Acausal.lead
```

## Retiming

Where `lag` and `lead` shift every row by one constant, `settime` recomputes
`:time` per row, from a column or a function.

```@docs
settime
```

The permissive variant, which may move rows earlier, is acausal and lives in
the `CausalFrames.Acausal` submodule. Unlike `futurejoin` and `lead` it is not
exported even from there, so that `using CausalFrames.Acausal` leaves the
causal `settime` unambiguous — reach it as `CausalFrames.Acausal.settime`.

```@docs
CausalFrames.Acausal.settime
```

## Summarization

```@docs
summarize
summarizecycles
intervalize
summarizewindows
addsummarycolumns
addrollingcolumns
Count
CountDistinct
Sum
SumPower
Moment
Product
DotProduct
Mean
Variance
Std
Covariance
Correlation
LinearRegression
Min
Max
First
Last
FitModel
```

## Model fitting (MLJ)

With an MLJ model package loaded (`using MLJ`), [`FitModel`](@ref) fits a model
inside any summarization transform, emitting [`FittedModel`](@ref)s; these
operators apply such models to a stream and read their diagnostics.

```@docs
FittedModel
applymodels
addpredictions
modelreports
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

## Internals

```@docs
CausalFrames.chunkmap
```
