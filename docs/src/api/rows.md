# Row transformations

Everything driven by one row at a time: testing a row, computing values from
it, and selecting or retiming rows. The column set either stays as it is or
grows by what a row function returns.

The forward-looking time shifts live in the `CausalFrames.Acausal` submodule
and are never re-exported — reach them with `using CausalFrames.Acausal`.

## `filterrows`

```@docs
filterrows
```

## `addcolumns`

```@docs
addcolumns
```

Neither of the truncating transforms is row-wise: both carry state across the
whole window, `head` a remaining-row budget and `lastrow` a store of rows per
key.

## `head`

```@docs
head
```

## `lastrow`

```@docs
lastrow
```

## `lag`

```@docs
lag
```

## `Acausal.lead`

`lead` shifts forward in time and so is acausal; it lives in the
`CausalFrames.Acausal` submodule and is not re-exported.

```@docs
CausalFrames.Acausal.lead
```

## `settime`

Where [`lag`](@ref "lag") and [`lead`](@ref "Acausal.lead") shift every row by
one constant, `settime` recomputes `:time` per row, from a column or a function.

```@docs
settime
```

## `Acausal.settime`

The permissive variant, which may move rows earlier, is acausal and lives in the
`CausalFrames.Acausal` submodule. Unlike `futurejoin` and `lead` it is not
exported even from there, so that `using CausalFrames.Acausal` leaves the causal
`settime` unambiguous — reach it as `CausalFrames.Acausal.settime`.

```@docs
CausalFrames.Acausal.settime
```
