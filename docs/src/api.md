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
```

## Sources

```@docs
emptyframe
clock
readcsv
```

## Row-wise transforms

```@docs
filterrows
addcolumns
```

## Summarization

```@docs
summarize
summarizecycles
addsummarycolumns
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
SummarizerState
CausalFrames.emptyvalue
CausalFrames.fresh
CausalFrames.update!
CausalFrames.value
CausalFrames.widenstate
CausalFrames.dependencies
```

## Internals

```@docs
CausalFrames.chunkmap
```
