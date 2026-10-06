# Sources

Contents: `emptyframe`, `concatenate`, `merge`, `clock`, `readtable`.

Sources start a pipeline, producing rows clipped to `[start, stop)`. The ones
here build, combine or lift in-memory streams; file readers are under
[File I/O](io.md).

## `emptyframe`

    emptyframe() -> CausalPipeline

A source producing no rows. Loading it gives a zero-row frame with only
`:time`. It is the identity of `concatenate` and `merge`.

## `concatenate`

    concatenate(ps::CausalPipeline...) -> CausalPipeline

A source running `ps` one after another over the same context and emitting
their rows end to end. Each pipeline starts only once the previous one is
exhausted, so a chain of file sources holds one file open at a time. With no
arguments it is `emptyframe`. To interleave rows by time instead, use
`merge`.

#### Arguments
- `ps`: the pipelines, in time order. Each is evaluated over the whole context
  and clips itself; keeping their data from overlapping is up to the caller.

Throws an `ArgumentError`, when the offending chunk arrives, if a pipeline's
column names differ from the first pipeline's, or if a pipeline's first time
precedes the previous one's last (equal times are allowed). Element types may
differ between pipelines; `DataFrame(frame)` promotes them.

```julia
dir = mktempdir()
jan, feb = joinpath(dir, "jan.csv"), joinpath(dir, "feb.csv")
scan(Context(0, 10), readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5])) |> writecsv(jan))
scan(Context(0, 10), readtable(DataFrame(time = [3], bid = [11.0])) |> writecsv(feb))

types = Dict(:time => Int, :bid => Float64)
p = concatenate(readcsv(jan; types), readcsv(feb; types))
DataFrame(load(Context(0, 10), p))

# output

3×2 DataFrame
 Row │ time   bid
     │ Int64  Float64
─────┼────────────────
   1 │     1     10.0
   2 │     2     10.5
   3 │     3     11.0
```

## `merge`

    merge(p::CausalPipeline, ps::CausalPipeline...; batchsize = 1024)
        -> CausalPipeline

A source running the pipelines side by side over the same context and
interleaving their rows by time. To join them end to end instead, use
`concatenate`. There is no zero-argument form; `emptyframe` is
the identity.

#### Arguments
- `p`, `ps...`: the pipelines. Each is evaluated over the whole context and
  clips itself. All run at once, so a merge of file sources holds every file
  open and buffers one chunk per input.

#### Keywords
- `batchsize = 1024`: the minimum number of rows per emitted chunk (the last
  may be smaller). Must be positive.

The output has the union of the inputs' columns: `:time`, then the others in
the order the pipelines introduce them. A row from a pipeline lacking a column
has `missing` there. A pipeline that produces no rows contributes no columns.
Rows at equal times come in argument order, and each pipeline's own row order is
kept.

A pipeline's column names are fixed by its first chunk; a later chunk with
different names, or the same names in a different order, is an
`ArgumentError`.

```julia
trades = readtable(DataFrame(time = [1, 3], price = [10.1, 10.3]))
quotes = readtable(DataFrame(time = [1, 2], bid = [10.0, 10.2]))
DataFrame(load(Context(0, 10), merge(trades, quotes)))

# output

4×3 DataFrame
 Row │ time   price      bid
     │ Int64  Float64?   Float64?
─────┼─────────────────────────────
   1 │     1       10.1  missing
   2 │     1  missing         10.0
   3 │     2  missing         10.2
   4 │     3       10.3  missing
```

## `clock`

    clock(interval; batchsize = 1024) -> CausalPipeline

A source with only a `:time` column, one row at each of `start`,
`start + interval`, … before `stop`.

#### Arguments
- `interval`: the tick spacing; anything that can be added to the time type
  (a `Dates.Period` for `DateTime`, a number for numeric time). Must be
  positive, checked when the pipeline runs.

#### Keywords
- `batchsize = 1024`: rows per emitted chunk. Must be positive.

## `readtable`

    readtable(table; time = nothing, checkorder = true, sort = false,
              closed = false, skipmissing = false) -> CausalPipeline
    readtable(frame::CausalFrame; closed = nothing, checkcontext = true)
        -> CausalPipeline

A source reading an in-memory table, clipped to `[start, stop)`. Only the rows
in the window are copied.

#### Arguments
- `table`: any Tables.jl table — a `DataFrame`, a `NamedTuple` of vectors, a
  vector of `NamedTuple`s. A table with several `Tables.partitions` is read one
  chunk per partition. A `DataFrame` is referenced, not copied, and is prepared
  when `readtable` is called, so its time errors are raised there; do not mutate
  it while the pipeline is in use.
- `frame`: a loaded `CausalFrame`, read back chunk by chunk. Its time is
  already resolved, so only `closed` and `checkcontext` apply.

#### Keywords
- `time = nothing`: where `:time` comes from, as for `readcsv`. A
  textual time column is an `ArgumentError`.
- `checkorder = true`: check that time is non-decreasing. With `false` the
  caller vouches for the order, and an unsorted table gives wrong results rather
  than an error.
- `sort = false`: stably sort the rows by time first. A partitioned table is
  concatenated to sort it.
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`. For
  a frame the default `nothing` keeps them exactly when the run's `stop` equals
  the frame's own, so `load(context(frame), readtable(frame))` reproduces
  `frame`.
- `skipmissing = false`: drop rows whose time is `missing`. Without it such a
  row is an `ArgumentError`.
- `checkcontext = true`: for a frame, require the run's context to lie within
  `context(frame)` (an `ArgumentError` otherwise); `false` clips the frame to any
  context.

```julia
df = DataFrame(ts = [3, 1, 2], bid = [1.0, 2.0, 3.0])
p = readtable(df; time = :ts, sort = true) |> filterrows(r -> r.bid > 1)
DataFrame(load(Context(0, 10), p))

# output

2×2 DataFrame
 Row │ time   bid
     │ Int64  Float64
─────┼────────────────
   1 │     1      2.0
   2 │     2      3.0
```
