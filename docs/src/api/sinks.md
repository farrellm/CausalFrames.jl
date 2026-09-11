# Sinks

Every sink is a pass-through transform: it writes each chunk as it flows by and
yields it downstream unchanged, so a sink can sit in the middle of a pipeline.
Pair one with [`scan`](@ref) to drive a pipeline for its side effects alone.

## `writecsv`

```@docs
writecsv
```

## `writeparquet`

```@docs
writeparquet
```

## `writejls`

```@docs
writejls
```
