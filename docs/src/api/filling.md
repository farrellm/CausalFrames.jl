# Filling and truncation

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

Neither of the truncating transforms is row-wise either: both carry state across
the whole window, `head` a remaining-row budget and `lastrow` a store of rows per
key.

## `head`

```@docs
head
```

## `lastrow`

```@docs
lastrow
```
