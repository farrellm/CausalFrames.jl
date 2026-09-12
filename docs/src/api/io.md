# File I/O

The operators that move rows between a pipeline and a file. Reading starts a
pipeline; writing does not end one — every writer is a pass-through transform,
so a sink can sit in the middle of a chain.

Parquet support is optional, through two backends: either `using DuckDB` or
`using Parquet2` enables both parquet operators. Reading prefers DuckDB (it
pushes the window into the reader) and writing prefers Parquet2 (it streams row
groups out); `backend = :duckdb` / `:parquet2` forces the choice.

## Reading

Every reader clips what it produces to the context's half-open interval
`[start, stop)`, and reads incrementally rather than loading the file.

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

## Writing

Every sink is a pass-through transform: it writes each chunk as it flows by and
yields it downstream unchanged. Pair one with [`scan`](@ref) to drive a pipeline
for its side effects alone.

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
