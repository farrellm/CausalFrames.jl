# Row transformations

Contents: `filterrows`, `addcolumns`, `head`, `lastrow`, `sortcycles`, `lag`, `warmup`, `Acausal.lead`, `settime`, `Acausal.settime`.

Transforms that filter, compute, truncate, reorder or retime rows. The
forward-looking `lead` and permissive `settime` are in the
`CausalFrames.Acausal` submodule (`using CausalFrames.Acausal`).

## `filterrows`

    filterrows(pred) -> (CausalPipeline -> CausalPipeline)
    filterrows(p::CausalPipeline, pred) -> CausalPipeline

A transform keeping the rows for which `pred(row)` is `true`.

#### Arguments
- `pred`: a function `row -> Bool`. `row` supports `row.name` and `row[:name]`,
  including `row.time`.

## `addcolumns`

    addcolumns(f) -> (CausalPipeline -> CausalPipeline)
    addcolumns(p::CausalPipeline, f) -> CausalPipeline

A transform appending columns computed from each row.

#### Arguments
- `f`: a function `row -> NamedTuple`, where the `NamedTuple` maps each new
  column name to its value in that row. `row` is as for `filterrows`.
  Returning anything but a `NamedTuple`, or a name already in the input
  (including `time`), is an `ArgumentError`.

```julia
p = readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7]))
DataFrame(load(Context(0, 10), p |> addcolumns(r -> (; mid = (r.bid + r.ask) / 2))))

# output

2×4 DataFrame
 Row │ time   bid      ask      mid
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     10.0     10.2     10.1
   2 │     2     10.5     10.7     10.6
```

## `head`

    head(n) -> (CausalPipeline -> CausalPipeline)
    head(p::CausalPipeline, n) -> CausalPipeline

A transform emitting the first `n` rows of the window, then stopping: its input
is not pulled again, so `readcsv(path; types) |> head(10)` reads a single file
chunk.

#### Arguments
- `n`: the number of rows, an `Integer`. Must be non-negative; `head(0)` never
  pulls its input.

`head` counts rows over the whole window, so `head(n)` over `[a, b)` and over
`[b, c)` can yield `2n` rows where `[a, c)` yields `n`.

Put writers downstream of `head`, never upstream (`p |> head(n) |>
writecsv(path)`): a writer finalizes its file only when its input is exhausted,
which `head` prevents.

## `lastrow`

    lastrow(; key = nothing) -> (CausalPipeline -> CausalPipeline)
    lastrow(p::CausalPipeline; key = nothing) -> CausalPipeline

A transform emitting the last row of the window, retimed to `stop`, with every
column kept. An empty input emits nothing. It emits once its input is
exhausted, so it streams as a single frame.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, emit each key's last row, sorted by key.

The original time is overwritten; keep it under another name if needed:

```julia
df = DataFrame(time = [1, 2, 3], sym = ["a", "b", "a"], px = [10, 20, 11])
p = readtable(df) |> addcolumns(r -> (; t0 = r.time)) |> lastrow(; key = :sym)
DataFrame(load(Context(0, 10), p))

# output

2×4 DataFrame
 Row │ time   sym     px     t0
     │ Int64  String  Int64  Int64
─────┼─────────────────────────────
   1 │    10  a          11      3
   2 │    10  b          20      2
```

A chunk whose column names differ from the first chunk's is an
`ArgumentError`.

## `sortcycles`

    sortcycles(by; rev = false) -> (CausalPipeline -> CausalPipeline)
    sortcycles(p::CausalPipeline, by; rev = false) -> CausalPipeline

A transform stably sorting the rows within each *cycle* — a run of rows sharing
one time — leaving the time order untouched. It holds back only the latest
cycle, so memory is one cycle plus one chunk.

#### Arguments
- `by`: the sort key — a column name, a non-empty collection of column names
  (compared lexicographically), or a function `row -> key`, with `row` as for
  `filterrows`. Keys compare with `isless`, so `missing` sorts last.
  Anything else is an `ArgumentError`, as is naming a column the data lacks.

#### Keywords
- `rev = false`: reverse the order. For a mixed order, negate a numeric key in
  a function instead: `sortcycles(r -> (-r.votes, r.id))`.

Sorting cycles turns a `Count` keyed on a copy of the time into a rank
(a key may not be `:time` itself):

```julia
films = DataFrame(year = [2020, 2020, 2020, 2021], id = [1, 2, 3, 4], votes = [5, 9, 9, 1])
p = readtable(films; time = :year) |>
    addcolumns(r -> (; year = r.time)) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :year)   # :count ranks films within a year
DataFrame(load(Context(2020, 2030), p))

# output

4×5 DataFrame
 Row │ time   id     votes  year   count
     │ Int64  Int64  Int64  Int64  Int64
─────┼───────────────────────────────────
   1 │  2020      2      9   2020      1
   2 │  2020      3      9   2020      2
   3 │  2020      1      5   2020      3
   4 │  2021      4      1   2021      1
```

## `lag`

    lag(offset) -> (CausalPipeline -> CausalPipeline)
    lag(p::CausalPipeline, offset) -> CausalPipeline

A transform moving every row `offset` later (`time -> time + offset`), so the
row at time `t` carries the values the input had at `t - offset`. Only `:time`
changes. The input runs over `[start - offset, stop - offset)`, so the output
fills the whole window.

#### Arguments
- `offset`: the shift, in a type that can be added to and subtracted from the
  time type (a `Dates.Period`, a number). Must be non-negative, checked when the
  pipeline runs; `0` is the identity. For a forward shift, see
  `Acausal.lead`.

## `warmup`

    warmup(lookback, f) -> (CausalPipeline -> CausalPipeline)
    warmup(p::CausalPipeline, lookback, f) -> CausalPipeline

A transform running `f(p)` over `[start - lookback, stop)` and dropping its
output rows before `start`, so a stateful transform enters the window already
warmed up instead of restarting at its first row. `f` sees the lead-in as input:
`summarize` folds it and `head` counts it.

If `lookback` covers everything `f` remembers, loading `[a, b)` and `[b, c)` and
concatenating the results equals loading `[a, c)`. That holds exactly for finite
memory (time windows, `forwardfill` with a `tolerance`, and `Bars` when
`lookback` spans the bars), and only to decay tolerance for a recursive state.
It never holds for a path-dependent state, such as a cumulative
`Sum` in `addsummarycolumns`, nor for output aligned to the
context's start, such as `clock` ticks.

Warm-ups compose by adding their lookbacks: for non-negative `x` and `y`,
`warmup(x, warmup(y, f))` equals `warmup(x + y, f)` whenever
`(start - x) - y == start - (x + y)`, which floating-point rounding and
calendar periods near a month's end can break.

#### Arguments
- `lookback`: how far before `start` `f` runs, in a type that can be subtracted
  from the time type (a `Dates.Period`, a number). Must be non-negative, checked
  when the pipeline runs; `0` and `nothing` are the identity.
- `f`: a function mapping a `CausalPipeline` to a `CausalPipeline`, such as a
  transform or a chain of them. Called once, when `warmup` is applied; any other
  return value is an `ArgumentError`.

```julia
df = DataFrame(time = [1, 4, 6], bid = [1.0, missing, missing])
p = readtable(df) |> warmup(5, forwardfill(:bid))
DataFrame(load(Context(3, 10), p))

# output

2×2 DataFrame
 Row │ time   bid
     │ Int64  Float64?
─────┼─────────────────
   1 │     4       1.0
   2 │     6       1.0
```

## `Acausal.lead`

    lead(offset) -> (CausalPipeline -> CausalPipeline)
    lead(p::CausalPipeline, offset) -> CausalPipeline

**Acausal.** The mirror of `lag`: moves every row
`offset` earlier (`time -> time - offset`), so the row at time `t` carries the
values the input had at `t + offset`. Only `:time` changes. The input runs over
`[start + offset, stop + offset)`, so the output fills the whole window.
Available after `using CausalFrames.Acausal`.

#### Arguments
- `offset`: the shift, in a type that can be added to and subtracted from the
  time type. Must be non-negative, checked when the pipeline runs; `0` is the
  identity.

## `settime`

    settime(spec) -> (CausalPipeline -> CausalPipeline)
    settime(p::CausalPipeline, spec) -> CausalPipeline

A transform recomputing `:time`. The result is converted to the context's time
type and clipped to `[start, stop)`; other columns pass through.

#### Arguments
- `spec`: either a `Symbol`, naming a column that replaces `:time` (taking its
  own position; the old `:time` is dropped), or a function `row -> time` whose
  result overwrites `:time` in place. `row` is as for `filterrows`.

Each of these is an `ArgumentError` when the pipeline runs: a row whose new
time is earlier than its old one (see
`Acausal.settime`), a new time column
that decreases within or across chunks, a textual new time, and a `missing` one
(drop such rows first with `filterrows`).

The input is not widened: it runs over `[start, stop)`, so a row outside the
window is never seen, even if `spec` would move it inside. Hence loading
`[a, c)` need not equal loading `[a, b)` and `[b, c)`. `settime(:time)` changes
no values, but still drops rows at `stop`.

## `Acausal.settime`

    CausalFrames.Acausal.settime(spec) -> (CausalPipeline -> CausalPipeline)
    CausalFrames.Acausal.settime(p::CausalPipeline, spec) -> CausalPipeline

**Acausal.** The permissive `settime`: the same
`spec`, conversion and clip to `[start, stop)`, but rows may move earlier. The
new time column must still be non-decreasing within and across chunks, and
neither textual nor `missing` (an `ArgumentError` otherwise).

Not exported even from `Acausal`, so that `using CausalFrames.Acausal` leaves
the causal `settime` unambiguous; call it as `CausalFrames.Acausal.settime`.

#### Arguments
- `spec`: a `Symbol` or a function `row -> time`, as for `settime`.

The input is not widened, so a row at or after `stop` is never seen, even if
`spec` would move it into the window. For a constant shift, use
`lead`, which does widen.
