# Summarizing transforms

Each of these folds one or more [summarizers](summarizers.md) over a set of
rows — the whole window, one timestamp, an interval, a trailing window — and an
optional `key` splits the fold per unique key value.

## `summarize`

```@docs
summarize
```

## `summarizecycles`

```@docs
summarizecycles
```

## `intervalize`

```@docs
intervalize
```

## `summarizewindows`

```@docs
summarizewindows
```

## `addsummarycolumns`

```@docs
addsummarycolumns
```

## `addrollingcolumns`

```@docs
addrollingcolumns
```
