# Sources

Sources start a pipeline, producing rows clipped to `[start, stop)`. The ones
here build, combine or lift in-memory streams; file readers are under
[File I/O](io.md).

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

## `readtable`

```@docs
readtable
```
