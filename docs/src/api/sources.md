# Sources

A source starts a pipeline: it produces rows rather than transforming them, and
clips what it produces to the context's half-open interval `[start, stop)`.
These construct or combine streams without touching a file — the file-backed
sources are under [File I/O](io.md).

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
