# Column transformations

The column set itself: which columns exist and in what order, and filling or
joining values into them. Rows pass through one for one — no row is added,
dropped, or retimed here.

The forward-looking join lives in the `CausalFrames.Acausal` submodule and is
never re-exported — reach it with `using CausalFrames.Acausal`.

## `selectcolumns`

```@docs
selectcolumns
```

## `dropcolumns`

```@docs
dropcolumns
```

## `reordercolumns`

```@docs
reordercolumns
```

Two ways to resolve `missing` values — the ones a [`merge`](@ref "merge") schema
union, an [`asofjoin`](@ref "asofjoin") non-match or a nullable parquet column
leaves behind. Only `fillmissing` is row-wise; `forwardfill` carries the last
non-missing value of every filled column across rows, chunks and keys.

## `forwardfill`

```@docs
forwardfill
```

## `fillmissing`

```@docs
fillmissing
```

A join appends another pipeline's columns alongside each row, matching by time.

## `asofjoin`

```@docs
asofjoin
```

## `Acausal.futurejoin`

```@docs
CausalFrames.Acausal.futurejoin
```
