# Column transformations

Contents: `selectcolumns`, `dropcolumns`, `reordercolumns`, `forwardfill`, `fillmissing`, `asofjoin`, `lookupjoin`, `Acausal.futurejoin`.

Transforms that choose, order, fill or join columns. Rows pass through one
for one, except that `lookupjoin` with `unmatched = :drop`
drops some. The forward-looking `futurejoin` is in the `CausalFrames.Acausal`
submodule (`using CausalFrames.Acausal`).

## `selectcolumns`

    selectcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    selectcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform keeping only the selected columns, in their input order. `:time` is
always kept.

#### Arguments
- `selectors`: at least one column selector: a name (`Symbol` or `String`),
  a `Regex` matched against names, a predicate called with the name as a
  `String`, or a collection of these. A column is kept if any selector matches
  it. A name the data lacks is an `ArgumentError`; a `Regex` or predicate
  matching nothing is not.

```julia
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> selectcolumns(:venue, r"^px_", startswith("qty"))
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "px_bid"
 "px_ask"
 "qty_bid"
 "venue"
```

## `dropcolumns`

    dropcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    dropcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform removing the selected columns and keeping the rest, in their input
order.

#### Arguments
- `selectors`: at least one column selector, as for
  `selectcolumns`. A column is dropped if any selector matches it.
  `:time` is never dropped: a `Regex` or predicate matching it is ignored, and
  naming it is an `ArgumentError`, as is naming a column the data lacks.

## `reordercolumns`

    reordercolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    reordercolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform moving the selected columns to the front, after `:time`, in the
order of the selectors. The other columns follow in their input order.

#### Arguments
- `selectors`: at least one column selector, as for
  `selectcolumns`. Nested collections are flattened in place. A `Regex`
  or predicate contributes its matches in input order, and a column matched
  twice goes where it was first matched. `:time` always stays first: a `Regex`
  or predicate matching it is ignored, and naming it is an `ArgumentError`, as
  is naming a column the data lacks.

```julia
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> reordercolumns(:venue, r"^qty_")   # time, venue, qty_bid, then the rest
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "venue"
 "qty_bid"
 "px_bid"
 "px_ask"
```

Filling replaces `missing` values, such as those left by a
`merge` or an unmatched `asofjoin`:
`forwardfill` with earlier values, `fillmissing` with constants.

## `forwardfill`

    forwardfill(selectors...; key = nothing, tolerance = nothing)
        -> (CausalPipeline -> CausalPipeline)
    forwardfill(p::CausalPipeline, selectors...; key = nothing,
                tolerance = nothing) -> CausalPipeline

A transform replacing `missing` in the selected columns with the column's last
non-missing value. Each column is filled independently. Rows before a column's
first value, or past `tolerance`, stay `missing`. To fill with a constant, use
`fillmissing`.

#### Arguments
- `selectors`: the columns to fill, as for `selectcolumns`. `:time` and
  key columns are never filled. The set of selected columns may not change
  between chunks.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, values carry only within the same key.
- `tolerance = nothing`: the maximum age of a carried value, measured from the
  row it came from; must be non-negative. The input then runs over
  `[start - tolerance, stop)`, so rows near `start` can be filled from before
  the window; the time type must support subtraction.

```julia
using Dates
t0 = DateTime(2026, 1, 1)
df = DataFrame(time = t0 .+ Minute.([0, 1, 2, 10]), sym = ["a", "b", "a", "a"],
               bid = [1.0, 2.0, missing, missing])
p = readtable(df) |> forwardfill(:bid; key = :sym, tolerance = Minute(5))
DataFrame(load(Context(t0, t0 + Hour(1)), p))

# output

4×3 DataFrame
 Row │ time                 sym     bid
     │ DateTime             String  Float64?
─────┼────────────────────────────────────────
   1 │ 2026-01-01T00:00:00  a             1.0
   2 │ 2026-01-01T00:01:00  b             2.0
   3 │ 2026-01-01T00:02:00  a             1.0
   4 │ 2026-01-01T00:10:00  a       missing
```

## `fillmissing`

    fillmissing(specs...) -> (CausalPipeline -> CausalPipeline)
    fillmissing(p::CausalPipeline, specs...) -> CausalPipeline

A transform replacing `missing` in the named columns with a constant per
column. A filled column's element type becomes
`promote_type(nonmissingtype(T), typeof(value))`; a column that cannot hold
`missing` is left alone. To carry the last value forward, use
`forwardfill`.

#### Arguments
- `specs`: at least one fill, as `name => value` pairs, a `NamedTuple`, or a
  collection of pairs. Names must be unique, may not be `time`, and must exist
  in the data (checked when a chunk arrives).

```julia
df = DataFrame(time = [1, 2], qty = [missing, 3.0], sym = ["a", missing])
p = readtable(df) |> fillmissing(:qty => 0.0, :sym => "")   # or ((qty = 0.0, sym = ""))
DataFrame(load(Context(0, 10), p))

# output

2×3 DataFrame
 Row │ time   qty      sym
     │ Int64  Float64  String
─────┼────────────────────────
   1 │     1      0.0  a
   2 │     2      3.0
```

Joins append columns from elsewhere: `asofjoin` from another pipeline,
matching by time, and `lookupjoin` from a table without time, matching by key.

## `asofjoin`

    asofjoin(right::CausalPipeline; key = nothing, tolerance = nothing,
             strict = false, leftprefix = nothing, rightprefix = nothing,
             righttime = nothing) -> (CausalPipeline -> CausalPipeline)
    asofjoin(left::CausalPipeline, right::CausalPipeline; ...) -> CausalPipeline

A transform joining each left row to the latest `right` row at or before its
time. Every left row is kept, with `right`'s non-time columns appended as
`Union{Missing, T}`: `missing` where no right row matches. Among right rows at
the same time, the last one wins. For a table with no time column, use
`lookupjoin`.

#### Arguments
- `right`: the pipeline to join from. If it produces no rows, left rows pass
  through unchanged (apart from `leftprefix`).

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present on both sides; a row matches only right rows with an
  equal key. Key columns appear once, from the left, never prefixed.
- `tolerance = nothing`: the maximum age of a match, `time - righttime <=
  tolerance`; must be non-negative. The right pipeline then runs over
  `[start - tolerance, stop)`, so rows near `start` can match earlier right
  rows; the time type must support subtraction. Without it, right runs over
  `[start, stop)` only.
- `strict = false`: match only right rows strictly before the left row.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique, so a self
  join needs a prefix.
- `righttime = nothing`: a name under which to keep the matched right row's
  time; by default it is dropped.

## `lookupjoin`

    lookupjoin(table; key, unmatched = :missing, leftprefix = nothing,
               rightprefix = nothing) -> (CausalPipeline -> CausalPipeline)
    lookupjoin(p::CausalPipeline, table; key, ...) -> CausalPipeline

A transform joining each row to the row of `table` with the same key, appending
`table`'s other columns. Having no time, `table` holds for the whole window.
To join data that changes over time, use `asofjoin`.

#### Arguments
- `table`: any Tables.jl table without a `:time` column (a `DataFrame`, a
  `CSV.File`, …), holding at most one row per key; a duplicate key is an
  `ArgumentError`. It is validated and copied when `lookupjoin` is called. An
  empty table still appends its columns.

#### Keywords
- `key`: required; a column name or collection of column names, present in both
  `table` and the input. Keys match by `isequal`, so `1` matches `1.0` and
  `missing` matches `missing`. Key columns appear once, from the input, never
  prefixed.
- `unmatched = :missing`: what to do with an input row whose key is not in
  `table` — `:missing` fills the appended columns with `missing` (so they become
  `Union{Missing, T}`), `:error` throws an `ArgumentError`, and `:drop` drops the
  row. Under `:error` and `:drop` the appended columns keep `table`'s types.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique.

```julia
trades = readtable(DataFrame(time = [1, 2], sym = ["a", "b"], qty = [100, 200]))
dim = DataFrame(sym = ["a", "b"], sector = ["tech", "energy"])
DataFrame(load(Context(0, 10), trades |> lookupjoin(dim; key = :sym)))

# output

2×4 DataFrame
 Row │ time   sym     qty    sector
     │ Int64  String  Int64  String?
─────┼───────────────────────────────
   1 │     1  a         100  tech
   2 │     2  b         200  energy
```

## `Acausal.futurejoin`

    futurejoin(right::CausalPipeline; key = nothing, tolerance = nothing,
               strict = false, leftprefix = nothing, rightprefix = nothing,
               righttime = nothing) -> (CausalPipeline -> CausalPipeline)
    futurejoin(left::CausalPipeline, right::CausalPipeline; ...) -> CausalPipeline

**Acausal.** The mirror of `asofjoin`: joins each
left row to the earliest `right` row at or after its time. Every left row is
kept, with `right`'s non-time columns appended as `Union{Missing, T}`:
`missing` where no right row matches. Among right rows at the same time, the
first one wins. Available after `using CausalFrames.Acausal`.

#### Arguments
- `right`: the pipeline to join from. If it produces no rows, left rows pass
  through unchanged (apart from `leftprefix`).

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present on both sides; a row matches only right rows with an
  equal key. Key columns appear once, from the left, never prefixed.
- `tolerance = nothing`: the maximum distance ahead, `righttime - time <=
  tolerance`; must be non-negative. The right pipeline then runs over
  `[start, stop + tolerance)`, so rows near `stop` can match later right rows;
  the time type must support addition. Without it, right runs over
  `[start, stop)` only.
- `strict = false`: match only right rows strictly after the left row.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique, so a self
  join needs a prefix.
- `righttime = nothing`: a name under which to keep the matched right row's
  time; by default it is dropped.

Right rows are buffered per key until a left row consumes or passes them, and
proving a key has no future match reads the rest of `right`, so memory is
O(right rows) in the worst case, against `asofjoin`'s O(keys).
