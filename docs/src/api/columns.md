# Column transformations

Transforms that choose, order, fill or join columns. Rows pass through one
for one, except that [`lookupjoin`](@ref "lookupjoin") with `unmatched = :drop`
drops some. The forward-looking `futurejoin` is in the `CausalFrames.Acausal`
submodule (`using CausalFrames.Acausal`).

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

Filling replaces `missing` values, such as those left by a
[`merge`](@ref "merge") or an unmatched [`asofjoin`](@ref "asofjoin"):
`forwardfill` with earlier values, `fillmissing` with constants.

## `forwardfill`

```@docs
forwardfill
```

## `fillmissing`

```@docs
fillmissing
```

Joins append columns from elsewhere: `asofjoin` from another pipeline,
matching by time, and `lookupjoin` from a table without time, matching by key.

## `asofjoin`

```@docs
asofjoin
```

## `lookupjoin`

```@docs
lookupjoin
```

## `Acausal.futurejoin`

```@docs
CausalFrames.Acausal.futurejoin
```
