# File I/O

Readers are sources. Writers are pass-through transforms, so they can sit
anywhere in a chain; end it with [`scan`](@ref) to run it for the file alone.

Parquet needs a backend, `using DuckDB` or `using Parquet2`; either enables both
operators. Reading prefers DuckDB, which pushes the window into the reader, and
writing prefers Parquet2, which streams row groups; `backend` overrides the
choice.

## Reading

Readers clip to `[start, stop)` and read incrementally.

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

Writers write each chunk on a background task as it flows by, and pass it on
unchanged.

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
