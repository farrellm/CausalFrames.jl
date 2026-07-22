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

## Joins

```@docs
asofjoin
```

## Summarization

```@docs
summarize
summarizecycles
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
