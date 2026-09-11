# Joins and time shifts

The transforms that align one stream to another, or move rows along the time
axis. The forward-looking ones live in the `CausalFrames.Acausal` submodule and
are never re-exported — reach them with `using CausalFrames.Acausal`.

## `asofjoin`

```@docs
asofjoin
```

## `Acausal.futurejoin`

```@docs
CausalFrames.Acausal.futurejoin
```

## `lag`

```@docs
lag
```

## `Acausal.lead`

`lead` shifts forward in time and so is acausal; like `futurejoin` it lives in
the `CausalFrames.Acausal` submodule and is not re-exported.

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
