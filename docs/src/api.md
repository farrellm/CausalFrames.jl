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
```

## Sinks

```@docs
writecsv
writeparquet
```

## Row-wise transforms

```@docs
filterrows
addcolumns
selectcolumns
dropcolumns
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
addsummarycolumns
addrollingcolumns
Count
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
