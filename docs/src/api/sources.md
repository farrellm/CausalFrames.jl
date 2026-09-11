# Sources

A source starts a pipeline: it produces rows rather than transforming them, and
clips what it produces to the context's half-open interval `[start, stop)`.

## `emptyframe`

```@docs
emptyframe
```

## `concatenate`

```@docs
concatenate
```

## `merge`

```@docs
Base.merge(::CausalPipeline, ::CausalPipeline...)
```

## `clock`

```@docs
clock
```

## `readcsv`

```@docs
readcsv
```

## `readparquet`

```@docs
readparquet
```

## `readjls`

```@docs
readjls
```
