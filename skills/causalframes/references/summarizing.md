# Summarizing transforms

Contents: `summarize`, `summarizecycles`, `intervalize`, `summarizewindows`, `addsummarycolumns`, `addrollingcolumns`, `Bars`.

These fold [summarizers](summarizers.md) over a set of rows: the whole
window, one time, an interval or a trailing window. An optional `key` splits
the fold by key value.

## `summarize`

    summarize(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)
    summarize(p::CausalPipeline, summarizers; key = nothing) -> CausalPipeline

A transform summarizing the whole window, emitted at `stop`, with columns
`time`, the key columns, then the summaries; the input columns are dropped.

#### Arguments
- `summarizers`: a `Summarizer` or a collection of them. Output names
  must be unique and may not be `time` or a key column.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key the output is one row, the summarizers' empty
  values for an empty input. With a key it is one row per key, sorted by key,
  and nothing for an empty input.

## `summarizecycles`

    summarizecycles(summarizers; key = nothing,
                    keyset = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizecycles(p::CausalPipeline, summarizers; ...) -> CausalPipeline

A transform summarizing each *cycle* — a run of rows sharing one time —
separately, emitting at the cycle's time with columns `time`, the key columns,
then the summaries; the input columns are dropped.

#### Arguments
- `summarizers`: a `Summarizer` or a collection of them.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, each cycle emits one row per key present, sorted by
  key.
- `keyset = nothing`: the key values, declared up front (requires `key`), as for
  `intervalize`. Every cycle then emits one row per declared key, in
  declared order, with empty values (`count = 0`, `mean = missing`) for keys
  without rows.

## `intervalize`

    intervalize(clock, summarizers; key = nothing, keyset = nothing,
                closelast = false) -> (CausalPipeline -> CausalPipeline)
    intervalize(p::CausalPipeline, clock, summarizers; ...) -> CausalPipeline

A transform summarizing the input over the intervals between consecutive clock
times `b₀ < b₁ < …`. The rows in each `[bₖ, bₖ₊₁)` are summarized and emitted
at `bₖ₊₁`, with columns `time`, the key columns, then the summaries; the input
columns are dropped. Rows before `b₀` are dropped.

#### Arguments
- `clock`: a pipeline whose `:time` column gives the boundaries (other columns
  are ignored), such as `clock`. An empty clock gives no output.
- `summarizers`: a `Summarizer` or a collection of them.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key every interval emits one row, an empty interval
  its summarizers' empty values (`count = 0`, `mean = missing`). With a key each
  interval emits one row per key present, sorted by key, and an empty interval
  emits nothing.
- `keyset = nothing`: the key values, declared up front (requires `key`): a
  collection of values for one key column, or of tuples or named tuples for
  several. Every interval then emits one row per declared key, in declared
  order, with empty values for keys without rows. Key columns take `keyset`'s
  types, and a row with an undeclared key is an `ArgumentError`.
- `closelast = false`: also emit the partial interval after the last boundary,
  at `stop`.

Empty values widen the output types (`Mean` gives `Union{Missing, Float64}`).

## `summarizewindows`

    summarizewindows(clock, lookback, summarizers; key = nothing,
                     keyset = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizewindows(p::CausalPipeline, clock, lookback, summarizers;
                     ...) -> CausalPipeline

A transform summarizing a trailing window at every clock tick: at each tick
`τ`, the rows with time in `[τ - lookback, τ)` are summarized and emitted at
`τ`, with columns `time`, the key columns, then the summaries; the input
columns are dropped. The input runs over `[start - lookback, stop)`, so the
first tick sees a full window. Unlike `intervalize`'s intervals,
windows may overlap or leave gaps.

#### Arguments
- `clock`: a pipeline whose `:time` column gives the ticks (other columns are
  ignored), such as `clock`.
- `lookback`: the window length; non-negative and subtractable from the time
  type.
- `summarizers`: a `Summarizer` or a collection of them.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key every tick emits one row, an empty window its
  summarizers' empty values. With a key each tick emits one row per key with
  rows in its window, sorted by key, plus one row of empty values for each key
  whose window has just emptied, so a downstream `asofjoin` sees it go
  empty.
- `keyset = nothing`: the key values, declared up front (requires `key`), as for
  `intervalize`. Every tick then emits one row per declared key, in
  declared order.

Each summarizer's window slides in O(1) per row when it is a
`GroupSummarizer`, in O(log window) per key per tick when it is another
`MonoidSummarizer`, and is re-folded in O(window) otherwise.

## `addsummarycolumns`

    addsummarycolumns(summarizers; key = nothing) -> (CausalPipeline -> CausalPipeline)
    addsummarycolumns(p::CausalPipeline, summarizers; key = nothing) -> CausalPipeline

A transform appending running summaries: each row gets the summary of every row
so far, itself included.

#### Arguments
- `summarizers`: a `Summarizer` or a collection of them. Their output
  columns may not collide with existing columns.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, each key keeps its own running summary, so a keyed
  `Count` numbers the rows of each key.

## `addrollingcolumns`

    addrollingcolumns(windows, summarizers; key = nothing, from = nothing,
                      sharedrun = true) -> (CausalPipeline -> CausalPipeline)
    addrollingcolumns(p::CausalPipeline, windows, summarizers;
                      ...) -> CausalPipeline

A transform appending, for each row at time `t` and each window, the summaries
of the rows with time in `[t - lookback, t]`, or of the last `n` rows under a
`Bars``(n)` look-back. Columns are named `{window}_{column}`, e.g.
`m5_price_sum`. The summarized pipeline runs over a context widened by the
longest time look-back, so the first row already sees a full window.

#### Arguments
- `windows`: window names and their look-backs, as a `NamedTuple`
  (`(m5 = Minute(5), h1 = Hour(1), b20 = Bars(20))`), a `name => lookback`
  pair, or a collection of pairs. Names must be unique; a time look-back must
  be non-negative and subtractable from the time type.
- `summarizers`: a `Summarizer` or a collection of them. Their output
  columns may not collide with existing columns.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present in both inputs. With a key, a row's window holds only
  rows with the same key.
- `from = nothing`: a pipeline to summarize instead of the input itself. Its
  rows relate to the output rows by time and key only. By default the input is
  summarized.
- `sharedrun = true`: without `from`, run the input once, over the widened
  context, and drop the rows before `start` from the output, so the output
  rows are the rows summarized. `false` takes the output rows from a run over
  the window itself, as a pipeline without this transform would give them; a
  time look-back then runs the input twice, once per context. The two agree
  without a time look-back, and on an input whose rows at or after `start`
  don't depend on earlier ones. Ignored with `from`.

A running count depends on where its run starts. The `w2` look-back widens
the context to start at 1, so by default the output shows the counts from
there, the ones summed:

```julia
p = clock(1) |> addsummarycolumns(Count())
t = addrollingcolumns((w2 = 2,), Sum(:count))
DataFrame(load(Context(3, 6), p |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      3             6
   2 │     4      4             9
   3 │     5      5            12
```

With `sharedrun = false` the output rows come from a run starting at 3, so
their counts restart at 1, while the summaries still sum the widened run's:

```julia
t = addrollingcolumns((w2 = 2,), Sum(:count); sharedrun = false)
DataFrame(load(Context(3, 6), p |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      1             6
   2 │     4      2             9
   3 │     5      3            12
```

An empty time window, including one for an unseen key, gives the summarizers'
empty values, so a column's type may widen (`Min` gives `Union{Missing, T}`).
A `Bars` window holding fewer than `n` rows gives `missing` instead.

## `Bars`

    Bars(n::Integer) -> Bars

A bar-count look-back for `addrollingcolumns`: the window at a row
holds the last `n` summarized rows with time at or before the row's (under
the row's key), where a time look-back holds the rows within a time span.
Until `n` rows have arrived, every output of the window is `missing`, even a
summarizer's whose empty value is not (`Count`, `Sum`), so
each of its columns is `Union{Missing, T}`.

Rows sharing a timestamp are admitted together, so a row followed by rows at
the same time has them in its window: this is the last `n` rows per row only
when times are unique per key. A bar count has no time span, so it doesn't
widen the summarized input to before `start`, and the first `n - 1` rows of each
key are `missing`. A time look-back in the same call does widen it, and the
`Bars` window then counts those earlier rows too.

#### Arguments
- `n`: the number of rows in the window; less than 1 is an `ArgumentError`.

```julia
df = DataFrame(time = [1, 2, 4, 7], x = [1.0, 2.0, 3.0, 4.0])
p = readtable(df) |> addrollingcolumns((b2 = Bars(2), t2 = 2), Sum(:x))
DataFrame(load(Context(0, 10), p))

# output

4×4 DataFrame
 Row │ time   x        b2_x_sum   t2_x_sum
     │ Int64  Float64  Float64?   Float64
─────┼─────────────────────────────────────
   1 │     1      1.0  missing         1.0
   2 │     2      2.0        3.0       3.0
   3 │     4      3.0        5.0       5.0
   4 │     7      4.0        7.0       4.0
```
