# CausalFrames.jl — Design

CausalFrames represents time-series tables: tabular data with a non-decreasing
`time` column. Data is described *lazily* as a pipeline and evaluated by
streaming chunks from operator to operator. A time window (`Context`) plus a
pipeline gives a `CausalFrame` via `load` (the only operation that holds the
whole window in memory), an iterator of frames via `stream`, or nothing via
`scan`, which runs the pipeline for its side effects.

```julia
using CausalFrames, Dates

p = readcsv("ticks.csv";
        types = Dict(:time => DateTime, :bid => Float64, :ask => Float64)) |>
    filterrows(r -> r.bid > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

frame = load(Context(DateTime(2026, 1, 1), DateTime(2026, 2, 1)), p)
```

## Core types

### `Context{T}`

A time window with fields `start::T` and `stop::T`, `start <= stop` enforced
at construction. `T` is anything ordered (`isless`): `DateTime`, `Date`, `Int`
ticks, `Float64` seconds, ….

### `CausalFrame{T}`

A materialized table, **opaque**: it is one or more time-disjoint DataFrame
chunks (the chunks a pipeline streamed, wrapped without copying), and users
never touch them directly.

Invariants, checked at construction:

- every chunk has a `:time` column with element type `<: T`;
- all chunks share their column names (element types may differ;
  `DataFrame(cf)` promotes on concatenation);
- time is non-decreasing within and across chunks;
- all times lie in the **closed** interval `[start, stop]` (see "Interval
  semantics").

The public constructors validate all of this, including an O(n) sortedness
scan, because they accept arbitrary DataFrames. `load` and `stream` use an
internal trusted constructor (the `Trusted` token) instead: the chunk protocol
already guarantees the invariants (sources validate their input, transforms
preserve order), so they keep only O(1)-per-chunk guards — cross-chunk order and
window bounds — which still catch a misbehaving hand-rolled source. Any other
construction site must use the validating path.

Public access:

- the Tables.jl interface, as a **column-access** table. `Tables.columns`
  materializes once, as a copy; rows are served by Tables.jl's row-view
  fallback over those columns, so touching both costs one materialization.
  `Tables.schema(cf)` needs no row scan (names from the first chunk, eltypes
  promoted across chunks) and matches `DataFrame(cf)`. `Tables.partitions(cf)`
  yields one copied partition per chunk; an empty frame yields the zero-row
  frame `DataFrame(cf)` would;
- `DataFrame(cf)`, which concatenates the chunks into a copy;
- `context(cf)`, `nrow(cf)`, `names(cf)`.

### `CausalPipeline`

A lazy description of how to produce data: conceptually a function
`Context -> single-pass lazy iterator of DataFrame chunks`, time non-decreasing
within and across chunks and no chunk empty. The run function's type is a
parameter (`CausalPipeline{F}`), never an abstract `Function` field. Nothing
runs until the iterator is consumed, by one of:

- `load(ctx, p) -> CausalFrame`: drains the iterator into a frame wrapping the
  chunks without copying. An empty result gives a zero-row frame with only
  `:time`.
- `stream(ctx, p) -> iterator of CausalFrames`: one frame per chunk (see
  "Causality and streaming").
- `scan(ctx, p) -> nothing`: drains the iterator, discarding every chunk, to
  run a pipeline for its side effects (such as `writecsv`).

Each also has a curried form (`load(ctx)` returns a function of the pipeline),
so a chain can end in its own evaluation: `p |> transform(args) |> load(ctx)`.
The two-argument form is primary. `load` and `scan` share the O(1) guards in
`checkchunk`; `stream` applies the same guards as it places sub-context
boundaries.

## Operators

- **Sources** take ordinary arguments and return a `CausalPipeline`.
- **Transforms** are curried: `filterrows(pred)` returns a
  `CausalPipeline -> CausalPipeline` function, so `source |> transform(args)`
  chains. Each also has a pipeline-first form, `filterrows(p, pred)`, a thin
  wrapper over the curried one.

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | no rows, only `:time` |
| `concatenate(ps...)` | source | the pipelines end to end, in time order, with identical columns (see "Concatenation") |
| `merge(ps...; batchsize)` | source | the pipelines interleaved by time, with the union of their columns (see "Merging") |
| `clock(interval; batchsize)` | source | rows at `start, start + interval, …` before `stop`, only `:time`, in chunks of `batchsize` |
| `readtable(table; time, checkorder, sort, closed, skipmissing)` / `readtable(frame; closed, checkcontext)` | source | an in-memory table or frame (see "Tables as sources") |
| `readcsv(path; types, time, rename, delim, sort, chunkbytes, closed, skipmissing)` | source | a CSV file, `String` columns unless typed, read in chunks of about `chunkbytes` bytes (see "Resolving the time column", "Missing times", "Sorting a file source") |
| `readparquet(path; time, rename, sort, closed, skipmissing, backend)` | source | a parquet file through DuckDB or Parquet2, skipping data outside the window (see "Parquet I/O") |
| `readjls(path; closed)` | source | a file written by `writejls` (see "JLS I/O") |
| `writecsv(path; queue, ...)` | transform | pass-through CSV sink (see "CSV output") |
| `writeparquet(path; queue, rowgroupsize, backend, ...)` | transform | pass-through parquet sink (see "Parquet I/O") |
| `writejls(path; queue)` | transform | pass-through `Serialization` sink (see "JLS I/O") |
| `filterrows(pred)` | transform | keep rows where `pred(row)` |
| `addcolumns(f)` | transform | append the `NamedTuple` `f(row)`, which may **not** contain `time` (so the time invariant needs no re-validation) |
| `selectcolumns(selectors...)` / `dropcolumns(selectors...)` / `reordercolumns(selectors...)` | transform | keep, drop, or move to the front the matching columns (see "Column selectors") |
| `summarize(ss; key)` | transform | the whole window, emitted at `stop` |
| `summarizecycles(ss; key, keyset)` | transform | each cycle (run of rows sharing a time) |
| `intervalize(clock, ss; key, keyset, closelast)` | transform | each interval `[bₖ, bₖ₊₁)` between clock ticks, at `bₖ₊₁` (see "Interval summarization") |
| `summarizewindows(clock, lookback, ss; key, keyset)` | transform | `[τ - lookback, τ)` at each clock tick `τ` (see "Window summarization") |
| `addsummarycolumns(ss; key)` | transform | append the running summary after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | append summaries over named trailing windows (see "Rolling windows") |
| `asofjoin(right; key, tolerance, strict, leftprefix, rightprefix, righttime)` | transform | append the latest right row at or before each row (see "As-of join") |
| `lookupjoin(table; key, unmatched, leftprefix, rightprefix)` | transform | append the row with the same key from a table without time (see "Lookup join") |
| `lag(offset)` | transform | move every row `offset` later (see "Lead and lag") |
| `settime(spec)` | transform | recompute `:time`; rows may only move later (see "Retiming") |
| `head(n)` | transform | the first `n` rows, then stop pulling (see "Truncation") |
| `lastrow(; key)` | transform | the last row per key, retimed to `stop` (see "Last row") |
| `sortcycles(by; rev)` | transform | stably sort the rows of each cycle (see "Sorting within a cycle") |
| `forwardfill(selectors...; key, tolerance)` / `fillmissing(specs...)` | transform | fill `missing` with the last value or a constant (see "Filling") |
| `applymodels(models; column, key, tolerance, strict, name, operation)` | transform | append predictions from the latest fitted model (see "Model fitting (MLJ)") |
| `addpredictions(clock, lookback, model, predictors, response; key, name, operation, verbosity)` | transform | refit on a trailing window at each tick and predict (see "Model fitting (MLJ)") |
| `modelreports(; column, name)` | transform | replace each fitted model with its report (see "Model fitting (MLJ)") |

Row functions (`pred`, `f`) receive a row supporting `row.name` and
`row[:name]`, including `row.time`. Transforms iterate the concretely typed
rows of a column table behind a per-chunk function barrier — never
`DataFrameRow`s, whose column access is type-unstable — so a row function
compiles to direct field access, like a summarizer's `update!`.

Names are lowercase, with no camelCase and no shadowing of Base functions
(`filter`, `empty`, `count`, `sum`, `join`).

## CSV output

`writecsv(path)` is a *transparent pass-through*: it writes each chunk as it
flows by and yields it downstream unchanged, so it can sit anywhere in a
chain, not only at the end. Combined with `scan` it persists a stream
without ever materializing the window.

Writing must not stall the pipeline on disk I/O, so it happens on a
background task (`Threads.@spawn`) fed by a bounded `Channel{DataFrame}` of
depth `queue` (default 1). The pipeline blocks only when the writer falls
more than `queue` chunks behind, and once at the end to join it. The writer
holds one file handle for the whole run and flushes after each chunk, so an
interrupted run still leaves a complete prefix on disk; `append` is false
only for the first chunk, which is what makes `CSV.write` emit the header
exactly once. The channel is `bind`ed to the task, so a writer failure
closes it with the exception and the pipeline task sees it at the next
`put!` rather than deadlocking on a full queue.

This is the one place chunk ownership is shared, and it needs care. A
consumer owns the chunk it is handed, and four operators use that licence to
mutate the chunk's *column index* in place (`asofjoin`'s `prefixleft!`, which
`lookupjoin` shares,
`addrollingcolumns`' `assembleempty`, `settime`'s symbol form, which drops the
old `:time` and renames another column onto it) — which would race the writer reading
the same DataFrame on another task. Column *vectors*, by contrast, are never
mutated in place anywhere: every operator builds new ones. So the writer
keeps the original chunk and downstream gets `DataFrame(c; copycols = false)`
— a private index over the same vectors, O(ncols) per chunk and nothing per
row.

The file is truncated when the run starts and finalized when the stream is
*exhausted*, via `chunkmap`'s once-only `flush`. Abandoning a `stream`
part-way therefore leaves the last chunks unwritten — `scan` is the entry
point to use when the file is the only thing wanted. A stream with no rows
yields an empty file, never a stale one — there is no chunk to take a header
from — and `readcsv` reads a zero-byte file back as an empty stream, so the
round trip holds. Keyword arguments pass through to
`CSV.write`, except `append`/`header`/`writeheader`/`partition`/`compress`,
which the transform controls itself and rejects eagerly.

## Parquet I/O

Parquet support is **optional**: DuckDB and Parquet2 are weak dependencies
behind package extensions, so a CSV-only user pays nothing for them, and CSV
itself stays a hard dependency, being both the flagship source and the backbone
of the precompile workload. **Either backend alone is enough** — each serves
both directions — but each direction has a preference, and `backend`
(`:auto`, `:duckdb`, `:parquet2`) forces the choice:

| Operator | Preferred | Fallback |
|---|---|---|
| `readparquet` | DuckDB: a filtered streaming query, one result chunk (≤ 2048 rows) at a time | Parquet2: one row group per chunk, skipped by its own statistics |
| `writeparquet` | Parquet2: one row group per `rowgroupsize` buffered rows, written as the stream flows by | DuckDB: chunks staged in a temp table, written by one `COPY` at the end |

The preferences follow what each library does well. Reading wants a *filtered*
scan: the context window becomes a `WHERE` clause on the time column, and
DuckDB skips the row groups **and pages** whose statistics put them outside it
(`DBInterface.prepare(con, sql, DuckDB.StreamResult)` plus
`Tables.partitions`). The Parquet2 reader gets the coarser half of that itself,
comparing each row group's recorded time min/max against the window
(`ColumnStatistics`) and skipping the group undecoded, or ending the scan once a
group starts at or past `stop`. Writing wants the opposite — an incremental
*writer* — which Parquet2 has and DuckDB does not: DuckDB cannot append row
groups to a parquet file, so its sink stages the stream in a temporary table
(spilled to its temp directory under memory pressure) and writes the file with
one `COPY`. That fallback therefore gives up the bounded-memory property the
Parquet2 sink shares with `writecsv`; `rowgroupsize` also becomes a hint there,
since DuckDB rounds it up to its 2048-row vector size.

Skipping never changes results, whichever reader does it. Rows are clipped
again on arrival by the same `clipchunk!` the CSV source uses, so a file whose
writer recorded no statistics, or one whose time column cannot be identified,
simply reads more of itself. Two cases fall back to a full scan by
construction: a `time` function, which is opaque to both readers (they still
stop at the first time past the window), and a `rename` that leaves no unique file
column mapping to `:time` — `timesourcename` returns `nothing` rather than
guess. As with `readcsv`, sortedness is only checked in the chunks actually
read, so a violation inside a skipped row group goes unreported. With
`closed = true` both skips move with the window: DuckDB's `WHERE` bounds the
time with `<=` rather than `<`, and a Parquet2 row group starting exactly at
`stop` is read rather than ending the scan — skipping it would be a wrong
answer, not a slower one. DuckDB's `WHERE` also drops null times, so without
`skipmissing` a null in a named time column goes unreported there while
Parquet2 raises it — error coverage, like sortedness, differs between the
readers, and successful results do not. Surfacing them with `OR col IS NULL`
was measured and rejected: the disjunction defeats the row-group skip, ~7x
slower on every default read even of a file with no nulls
(`notes/duckdb-null-pushdown.md`).

Backend selection resolves twice: eagerly at construction, so a missing backend
is reported where the operator was typed, and again inside `run(ctx)`, so a
backend loaded after the pipeline was built still counts. The run-time
resolution dispatches on a runtime symbol — one dynamic call per pipeline run,
not per chunk — and hands a `Val` to `parquetproducer` / `parquetsink`, whose
per-backend methods live in the extensions.

One in-memory DuckDB database is created lazily per process and a connection
opened per `run(ctx)` — a connection is single-consumer, and a pipeline may be
run more than once (a self-join reads its file twice).

Both sinks reuse the CSV sink's machinery whole: the shared `ChunkSink` (bounded
`Channel{DataFrame}`, background task, `bind`ed so a writer failure surfaces at
the next `put!`, the first-chunk column check, and the
`DataFrame(c; copycols = false)` hand-off that keeps the writer's chunk private
from downstream index mutation). Only the write loop differs. Holding chunks —
to fill a row group, or to stage the whole output — is safe under the same
contract: column *vectors* are never mutated in place anywhere, only replaced
wholesale. Chunks are only ever merged, never split, so `rowgroupsize = 1`
writes one row group per incoming chunk.

The one semantic difference from `writecsv`: **a parquet file is only valid
once finalized**. The footer (or, for DuckDB, the whole file) is written when
the stream is exhausted, so there is no usable prefix on disk mid-run and
abandoning a `stream` part-way leaves an unusable file — `scan` is the entry
point when the file is the point. Both sinks truncate the file when the run
starts, as `writecsv` does — the DuckDB sink explicitly, since otherwise it
touches the file only at the final `COPY`, and a failed run would leave the
previous one's file looking current. A stream with no rows yields a valid file
of zero rows and one `time::Int64` column under either sink (DuckDB cannot read
a parquet file with no columns at all), which both readers return as an empty
stream. Keyword arguments pass through to `Parquet2.FileWriter`, where
`compute_statistics` defaults to `["time"]` so that files written here carry the
statistics the readers skip by; the DuckDB sink understands `compression_codec`
(mapped onto `COPY`'s `COMPRESSION`, and it records statistics of its own) and
rejects the other, Parquet2-specific options rather than silently dropping them.

## Sorting a file source

`readcsv` and `readparquet` take `sort = true` for files not stored in time
order — a parquet file written by a query without `ORDER BY` is the usual case.
It has `readtable`'s meaning: a stable sort by time, since rows sharing a
timestamp form a cycle and their order within it is data, applied at the source
where it costs no causality. The source's output obeys every invariant; only the
reading changes.

A sort cannot stream, so everything that relies on the file's order is dropped
from that read: the order check, the binary-searched clip and the early stop at
the first time `>= stop`. `gatherchunk!` is `clipchunk!`'s counterpart for it —
the same `renamecolumns!`/`resolvetime!`, then a scan (`windowrows`, typed on the
raw time vector) keeping the in-window rows of each file chunk. Once the file is
exhausted, `sortgathered` concatenates them, applies `stableperm` unless they are
already in order, and emits the lot as **one chunk**. Memory therefore scales
with the rows in the window, plus one file chunk, never with the file. Chunks
are gathered in file order, so stability across them is file order too.

Where a reader can do better, it does. DuckDB takes the sort into the query as
`ORDER BY <time>, file_row_number` beside the window's `WHERE`, and the sorted
result streams through the ordinary `clipchunk!` loop, early stop included;
DuckDB spills a large sort to disk itself. `file_row_number` is there because
DuckDB's `ORDER BY` is not stable: it is the reader's virtual column, left out of
`*`, and the tiebreak that makes the order the file's. Pushdown needs a nameable
time column and that tiebreak, so a `time` function, an untraceable `rename`, or
a file with a column of its own named `file_row_number` (which shadows the
virtual one) sends DuckDB down the gather-and-sort path instead. Parquet2 always
gathers; its statistics still skip row groups wholly outside the window, but
under a sort a group at or past `stop` no longer ends the scan.

A `sort = true` read raises no order error, and none is checked separately: the
output is sorted by construction. The within-timestamp half of an `ORDER BY` —
secondary sort keys — is not a source option, here or in `readtable`: it needs no
source, since reordering rows that share a timestamp never moves one in time, and
so it is the transform `sortcycles` (see "Sorting within a cycle").

## Resolving the time column

The three sources that resolve their own times — `readcsv`, `readparquet` and
`readtable` over a table or `DataFrame` — share one set of rules, whether the
work is done by `resolvetime!` (file chunks) or `tabletimes` (tables):

- `time` is `nothing`, a `Symbol` or a function, checked when the operator is
  called (`checksourcetimespec`). The chunk path acts only on a `Symbol` or a
  function, so anything else would otherwise quietly read the `:time` column.
- `time = :name` when the source also has a `:time` column is an error, never a
  silent overwrite: which column the user meant is ambiguous.
- A textual time cannot be ordered against the window and is an error, whether
  it came from a column or from a `time` function. The message names the way
  out: fix the function, or (CSV only) type the column through `types`.
- Data errors name their source the same way — "CSV file x.csv", "parquet file
  x.parquet", "jls file x.jls", or "table" — through `sourcename`, and every
  message is built only on the error path, never per chunk.

## Missing times

A time that is `missing` has no place on the time axis, so every source that
resolves its own times — `readcsv`, `readparquet`, `readtable` over a table or
`DataFrame` — refuses one by default, with an error naming the way out:
`skipmissing = true` drops those rows instead. Where the missing times come
from does not matter: a blank cell in a typed CSV time column, a null in a
parquet one, a `Union{Missing,T}` table column, or a `time` function returning
`missing`. A frame, and so `readjls` and `readtable(frame)`, never holds one,
its time eltype being `<: T`.

The check sits right after the time is resolved and before everything that
orders it — the order check, the clip, the sort — in one typed barrier,
`presentrows`, shared by `clipchunk!`, `gatherchunk!` and both `readtable`
paths. It returns `nothing` unless a time is missing — decided on the eltype
alone when it admits no `Missing`, so a typed column pays nothing — and
otherwise the indices of the present rows, which the clip composes with its
window (`present[lo:hi]`) so a chunk holding some missing times costs one copy
of its in-window rows, never a delete-then-slice. As with sortedness, a
missing time is only detected in the chunks actually read: one past the early
stop, or in a skipped row group, goes unreported.

Without the check a missing time fell through to whatever it broke: `missing`
sorts last, so mid-chunk it failed the order check with a misleading message,
at the end it read as a time past the window and was silently dropped by the
early stop, and in a `DataFrame` it reached a boolean test as a `TypeError`.

## JLS I/O

`writejls` and `readjls` persist a stream through Julia's `Serialization`
stdlib. They exist because the other two formats are *typed*: CSV writes a
value's printed form and parquet encodes only its own column types, so a column
holding arbitrary Julia values — a fitted model, a `NamedTuple` — has no round
trip through either. A JLS file stores whatever the chunks hold. `Serialization`
is a stdlib already in the sysimage, so this costs no dependency weight.

The format is a header record `(format = :CausalFramesJLS, version = 1)`
followed by one serialized `DataFrame` per chunk. Each record is its own
`serialize` call, so records carry no back-references to one another and the
reader can `deserialize` them one at a time. The header is a `NamedTuple` of
isbits values, which serializes identically across Julia versions, so a foreign
file or a future format version is reported as such rather than as an opaque
deserialization failure.

The sink is the shared `ChunkSink` whole — background task, bounded queue,
first-chunk column check, the `copycols = false` hand-off — with a write loop
that serializes and flushes each chunk. Like `writecsv`, and unlike the parquet
sinks, it therefore has a usable prefix mid-run; a stream with no rows leaves a
header-only file, which reads back as an empty stream. The source is a
`CSVProducer`-shaped `JLSProducer` that deserializes one record per pull and
hands it to the shared `clipchunk!` (no `time` or `rename` — the file was
written from a stream, so its `:time` is already resolved), stopping at the
first time past the window. There is no index to seek by, so a read costs the file's
prefix up to `stop`, as CSV's does.

Three caveats, all of them `Serialization`'s: a file is readable only by a
compatible Julia and compatible versions of the packages whose types it holds;
deserialization can construct arbitrary types, so a file must be trusted; and
an interrupted run leaves every complete record readable while its torn last
record is reported as an `ArgumentError` rather than a bare `EOFError`. JLS is a
persistence format for a pipeline's own outputs, not an interchange format —
`writecsv` and `writeparquet` remain that.

## Tables as sources

`readtable` lifts in-memory data into a pipeline. It is a source like any other
— each run clips to the context's `[start, stop)` and converts `:time` to the
context's time type — on three paths that share one clip rule (`windowbounds`)
and differ in *when* the per-table work runs and in what is copied.

- **Any Tables.jl table.** Each run walks `Tables.partitions(table)` through a
  `CSVProducer`-shaped `TableProducer`, one chunk per partition. Per partition,
  `tabletimes` resolves the time — `time` works as for `readcsv`: `:time`, a
  column renamed where it stands, or a per-row function behind `maptime` — and
  `tablerows`, the function barrier typed on the time vector and the context's
  time type, checks the order (within the partition and against the previous
  one's last time), binary-searches the window, and returns the rows to keep.
  Only those rows are copied, one column at a time, so a narrow window over a
  large table costs the window rather than the table, and a loaded frame never
  aliases the caller's vectors — which the caller may still mutate, since a
  pipeline is lazy and may run more than once. As for the file sources, reading
  stops at the first partition holding a time past the window.
- **An `AbstractDataFrame`.** The same resolution, done once when `readtable` is
  called, into a private column index over the caller's vectors
  (`DataFrame(df; copycols = false)` — O(ncols), and the rename never reaches
  `df`). A run is then the frame path over that one chunk: two binary searches
  and one `getindex`, with no per-run order scan, time function or `Tables.jl`
  dispatch, which matters most for pipelines run many times (a self join, a
  `stream`, `addrollingcolumns(; from)`). Errors the generic path raises at run
  time are raised by `readtable` itself. The copy rule survives: while the chunk
  aliases the caller's vectors every run slices, even over the whole window, and
  only once a `sort` has permuted the rows into copies of its own do
  whole-window runs share them. The price of resolving eagerly is that mutating
  the DataFrame's values while the pipeline is in use is unsupported — the order
  check has already run.
- **A `CausalFrame`.** Its chunks already satisfy every invariant, so nothing is
  resolved or checked. A binary search over the chunks' first and last times
  picks those that meet the window, and `clipframechunk` hands each on — a chunk
  wholly inside as `DataFrame(c; copycols = false)`, a private index over the
  frame's own vectors, a partial one as a slice. Sharing is sound for the reason
  the CSV sink's hand-off is: consumers may mutate a chunk's column index, but no
  operator mutates a column vector in place, and the frame never exposes its
  chunks. The frame method accepts only `closed` and `checkcontext`; `time`,
  `sort` and `checkorder` would have nothing to act on.

Two rules are particular to frames. A frame knows its rows only over its own
context: outside it the data is *unknown*, not absent, and yielding no rows
there would pass missing data off as an empty stretch — so a run over a context
not within `context(frame)` is an `ArgumentError`, and `checkcontext = false`
clips to whatever the frame holds instead. And a frame may hold rows exactly at
its `stop` (`summarize` emits there), which the half-open clip would drop:
`closed = nothing`, the frame default, closes the window exactly when the run's
`stop` equals the frame's, so `load(context(f), readtable(f))` reproduces `f`,
while every narrower window stays half-open and so still tiles.

`closed = true` opts in to `[start, stop]`, as it does on the file sources (see
"Interval semantics"), legal because frames tolerate the closed interval. `sort = true` sorts stably (rows sharing a
timestamp are a cycle, and their order within it is data); a partitioned table is
concatenated to sort it, and a table already in order skips the permutation.
`checkorder = false` skips the order scans: the caller vouches for the order, and
`load`'s O(1) `checkchunk` guards are all that remain.

## Column selectors

`selectcolumns` and `dropcolumns` project a stream onto a subset of its
columns, and `reordercolumns` permutes them. All three are variadic, and each
selector is one of:

- a column name — a `Symbol` or an `AbstractString`;
- a `Regex`, matched against the column name with `occursin`;
- a predicate, called with the column name as a `String` (the DataFrames
  `Cols(f)` convention, and what makes `startswith("px_")` work directly);
- recursively, any collection of those.

For the two **projections**, a column matches when *any* selector matches it;
`selectcolumns` keeps the matches and `dropcolumns` keeps the rest, both in the
**input's own column order**, never the selectors'. Selecting nothing is legal
(the result is a `:time`-only stream); a projection that keeps every column
passes its chunks through untouched.

`reordercolumns` is the one operator that reads the selectors as an *order*.
It moves the matching columns to the front in the **selectors' own order**,
every unmatched column following in the input's order — so a reorder names only
the columns it cares about and never has to re-list the schema. Three rules
follow from ordering by the selectors rather than matching against them:

- nested collections are **flattened in place**, so `reordercolumns([:a, :b])`
  orders as `reordercolumns(:a, :b)` does. A collection is punctuation, not a
  group;
- a `Regex` or predicate contributes every column it matches, among themselves
  in the input's order — a pattern says which columns, not which order;
- a column matched by more than one selector is placed by the **first** of
  them, and appears once.

A reorder that asks for the order the chunk already has passes it through
untouched, exactly as an all-keeping projection does.

- **`:time` is implicit.** It is always kept, whatever the selectors say, and
  a `Regex` or predicate matching `"time"` is ignored rather than obeyed.
  Naming it outright in `dropcolumns` is an `ArgumentError`, raised eagerly
  at construction — the `addcolumns` rule that a row function may not return
  a `time` key, from the other direction. `reordercolumns` pins it **first**
  and rejects naming it the same way: it is the one column whose position is
  not the caller's to choose, so a request to place it elsewhere would be a
  lie rather than a no-op.
- **A named column absent from the data is an `ArgumentError`**, so a typo
  fails rather than silently selecting nothing. A `Regex` or predicate
  matching nothing is not an error. Requesting zero selectors, or a selector
  that is neither a name, a pattern, a predicate, nor a collection, is an
  `ArgumentError` at construction.

The work is per column, not per row: a chunk is projected — or permuted — with
`df[!, keep]`, which shares the column vectors rather than copying them (the
chunk is owned, as in `addcolumns`). Resolving the selectors is memoized per
run against the column names it was resolved from, so a stream whose schema
never moves — the norm — runs the selectors and the validation once, on its
first chunk, while a schema that does move is re-resolved and re-validated
rather than projected through a stale column list.

Ordering needs one primitive the projections do not: matching asks only whether
*any* selector matched, so it can walk the spec in any order and stop early,
while `reordercolumns` must visit every leaf, in the order written. Hence
`foreachselector` beside `foreachliteral` — the same recursion, but visiting
the pattern and predicate leaves rather than skipping them.

## Truncation

`head(n)` emits the first up to `n` rows of the stream and then stops. Stopping
is the whole point, and it is why `head` is the one transform not built on
`chunkmap`: `chunkmap`'s `advance` loops until upstream returns `nothing`, so a
`step` returning `nothing` only *skips* a chunk. A `head` built that way would
still read the entire file.

Instead `head` returns a `ChunkSource` over a mutable `HeadProducer`, which
drives the upstream iterator itself — readcsv's `CSVProducer` shape, with the
pull-to-pull state in fields rather than in reassigned closure captures (which
would be boxed). `ChunkSource.produce()` takes no arguments, so the upstream
iterator *and* its state must be fields; the dynamically typed `state` field is
the same trade `ConcatProducer` already makes, and it costs one dynamic
`iterate` dispatch **per chunk**, never per row. The pulled chunk carries a
`::DataFrame` annotation — the chunk protocol guarantees it — so `produce()`
still infers `Union{Nothing, DataFrame}` through that field.

A chunk that fits entirely under the remaining budget is passed on untouched
(the chunk is owned and column vectors are never mutated in place, so the
"keeps everything, so no copy" rule `filterchunk` and `clipchunk!` follow
applies); only the chunk that spends the budget is sliced. `head(0)` builds the
upstream pipeline but never iterates it, so a generator source is never
advanced. `n` must be non-negative, rejected eagerly at construction.

`head` is **causal** — rows pass through unchanged and in order — but
**stateful**: its row budget spans the whole window. Concatenating the frames of
`stream(ctx, p |> head(n))` therefore still equals `load(ctx, p |> head(n))`
(`stream`'s one chunk of lookahead calls `produce()` once more, which returns
`nothing` without touching upstream), while the chunk-concatenation property
over *split* contexts does not hold: `head(n)` over `[a, b)` and over `[b, c)`
yields up to `2n` rows where the whole window yields `n`. It is stateful in
exactly the sense `summarize` is.

One known limitation. The sinks finalize in `chunkmap`'s once-only `flush`, and
`head` abandons its upstream, so a sink placed *upstream* of `head` never
finalizes: its channel is never closed, its writer task blocks forever holding
an open file, and the file is left unfinished. This is the documented
"abandoning a `stream` part-way" hazard, but `head` makes it reachable from a
fully drained `load`, which is new. Truncate first —
`p |> head(n) |> writecsv(path)`, never the reverse. Fixing it properly wants a
`close`-style hook on the chunk protocol so an abandoning consumer can release
its upstream; that does not exist today and is deliberately out of scope here.

## Concatenation

`concatenate(ps...)` is the **n-ary** combinator: it runs the pipelines one
after another over the same context and emits their chunks end to end, the
time-wise concatenation of their outputs. It is not curried — every argument
is a `CausalPipeline`, so a curried form would be indistinguishable from a
direct one, and there is no uncurried/curried pair to provide. Zero pipelines
gives `emptyframe()`, the identity of concatenation.

Two preconditions, both the caller's to satisfy and both checked as the chunks
flow by (O(ncols) per chunk, nothing per row):

- **Time order.** The pipelines must be passed in time order; a chunk whose
  first time precedes the last time already emitted is an `ArgumentError`
  naming the two pipelines. Equal times across a boundary are fine — the
  output only has to be non-decreasing. Within one pipeline the chunk protocol
  already guarantees this, but checking every chunk costs nothing extra and
  turns what `load`'s `checkchunk` would report as a generic cross-chunk
  violation into a message that names the pipeline that is out of place.
- **Identical columns.** Chunks must carry the same column names in the same
  order as the first chunk seen, the same rule the sinks apply in
  `sinkchunk`. Element *types* may still differ between pipelines, exactly as
  they may between the chunks of one pipeline: `DataFrame(cf)` promotes on
  concatenation.

There is no interleaving here: the point of `concatenate` is that the streams
are already disjoint in time, so no pipeline has to be looked at until the
previous one is done. Streams that do overlap are `merge`'s business (see
"Merging"), which pays for the interleaving with a chunk of lookahead per
pipeline. Each pipeline is evaluated over the **whole** context
`[start, stop)` and clips itself, so overlapping windows are caught by the
time-order check rather than silently reordered.

The mechanism is the `ChunkSource` producer `ConcatProducer`, in the shape of
readcsv's `CSVProducer`: per-run state in fields rather than reassigned
closure captures, and the dynamically typed fields (the current chunk
iterator and its state) touched once per *pipeline*, not per chunk and never
per row. A pipeline's `run(ctx)` is called only once the previous one is
exhausted, which keeps the chain as lazy as its parts — a chain of file
sources holds one file open at a time.

Causality is trivial: rows pass through unchanged and in time order, so
output at time `t` still depends only on input rows at time `≤ t`.

## Merging

`merge(ps...)` is the other **n-ary** combinator, and the counterpart to
concatenation: it runs the pipelines *concurrently* over the same context and
interleaves their rows by time. Like `concatenate` it is a source and is not
curried, for the same reason — every argument is a `CausalPipeline`. Unlike
`concatenate` there is no zero-argument form: a bare varargs method would also
capture the `merge()` call, and `merge` is `Base.merge`, extended here for our
own type rather than shadowed by a new name (the arguments are all
`CausalPipeline`s, so this is a method extension, not piracy). `emptyframe()`
is the identity of merging; one pipeline is that pipeline.

- **Union schema.** The pipelines need not have the same columns. The output
  carries `:time` first and then their columns in the order the pipelines
  introduce them; a row from a pipeline that lacks a column carries `missing`
  there, so a merged column of `T` materializes as `Union{Missing, T}`. A
  pipeline's own names are fixed by its first chunk, and a later chunk that
  renames or merely reorders them is an `ArgumentError` naming that pipeline —
  concatenate's `checkconcat!` rule. A pipeline producing no chunk at all over
  the window contributes no columns, exactly as an empty right stream
  contributes none to an as-of join: schemas are data-driven everywhere here,
  and with no chunk there is no schema to take.
- **Order.** The emission key is `(time, argument index)`, lexicographic, so
  rows at equal times go out in argument order — all the tied rows of the
  first pipeline, then the second's — and each pipeline's own row order is
  preserved. This is the same tie convention as concatenate's "equal times
  across a boundary are fine, stream order decides".

The union schema has to be settled before the *first* chunk goes out, because
a frame's chunks must all carry the same column names. So the first pull runs
every pipeline and buffers one chunk from each — which the merge needs anyway
to know whose row comes first, so nothing is pulled that the first block would
not have pulled. Thereafter each pipeline is one chunk ahead at most. That is
the cost of interleaving: `merge` holds `n` sources open at once and `n`
chunks resident, where `concatenate` holds one.

The mechanism is a `ChunkSource` producer over one `MergeCursor` per pipeline
— the buffered chunk, the row reached in it, and the lazily refilled iterator
behind it, in the shape of asofjoin's right stream. The cursors sit in a
`Vector{MergeCursor{T}}` rather than a tuple: selection indexes them by a
runtime index, and the only field the ordering touches (`times`, the chunk's
time column as the context's time type) is already concrete in the flat
struct.

Rows are claimed a **piece** at a time, never a row at a time. Each step picks
the cursor with the smallest key and claims from it the longest run of rows
that stays below the runner-up's key — one
`searchsortedfirst`/`searchsortedlast` on an already-sorted vector, and no data
movement at all: a piece is just the chunk, a row range into it, and the
cursor's union-position-to-column map. The winner and the runner-up are
distinct cursors and equal head times imply the winner has the smaller index
(it would not have won otherwise), which is what makes the run non-empty and
every step productive.

Pieces accumulate until `batchsize` rows are claimed (`clock`'s knob, same
default) and are then materialized together, one allocation per output column,
each row copied exactly once. Without the batching, two streams alternating row
by row would emit one chunk per row and `load` would build a vector of chunks
as long as the data; without the deferral, those rows would be copied twice,
once into a per-piece frame and once more to concatenate them. The column's
element type is the promotion of the pieces' own — `Missing` among them
wherever a piece's input lacks the column, which is what widens it to
`Union{Missing, T}`. The type is a runtime value, so the filling sits behind a
function barrier where the output column is concrete; the offsets ride in a
reused mutable `CopySpan` rather than as loose `Int` arguments, because the
source column's type is known only at run time and a dynamic call boxes every
non-pointer argument it is passed.

A batch that is a single piece covering a whole buffered chunk skips the copy
entirely: the chunk is owned and column vectors are never mutated in place, so
its columns are adopted as they are, and a chunk that already carries the union
schema goes downstream untouched. A pipeline that dominates a stretch of time
therefore passes through free of charge — only genuine interleaving copies.

Causality holds in the ordinary sense: an output row at time `t` carries
exactly one input row's values at `t`, plus `missing` in the columns that
pipeline does not have. The lookahead is at other cursors' head *times*, which
are `≥ t`, and it decides ordering only — never a value. Chunk concatenation
holds row for row, with the one caveat it shares with an as-of join's empty
right stream: a pipeline with no rows in a sub-window contributes no columns
*there*, so merging two halves of a window separately can give two frames with
different schemas even though their rows concatenate correctly.

## As-of join

`asofjoin(right; ...)` is the first **binary** operator: the curried
transform closes over a second pipeline, so `left |> asofjoin(right)` joins
two streams. Each left row is joined to the most recent right row whose time
is not after the left row's time (`strict = true`: strictly before). Every
left row is kept; the right table's value columns are appended with element
type `Union{Missing, T}` and are `missing` where no right row qualifies.
Among right rows sharing one time, the last in stream order wins. The right
`time` column is dropped unless `righttime` names an output column to
receive the matched row's time.

- **Keys.** With `key` (a column name or collection, present in both
  tables — validated on each side's first chunk) rows join per unique key
  value, exact-matched with `isequal`. The key columns appear once in the
  output, taken from the left row, never prefixed. `time` may not be a key.
- **Tolerance.** With `tolerance` a match additionally requires
  `time - rtime <= tolerance` (inclusive; checked per left row against the
  stored right row, never by eager eviction). The right pipeline then runs
  over the widened context `[start - tolerance, stop)` so lookback near the
  window start is fully covered — the only place the time type needs
  subtraction (`T - tolerance` yielding a time, `T - T` comparable to
  `tolerance`; numbers and `Dates` types qualify). Without `tolerance` the
  right pipeline sees only `[start, stop)`, so left rows near `start` may
  find no earlier right row. Negative tolerance is rejected at run time,
  generically, via `start - tolerance <= start` (the `clock` precedent).
- **Prefixes.** `leftprefix` / `rightprefix` rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique after
  prefixing — checked once, when both schemas are first known — so a
  no-prefix self join fails deterministically.
- **Self join.** `p |> asofjoin(p; rightprefix = "prev", strict = true)`
  gives previous-row semantics. It works because pipelines are lazy: each
  `run(ctx)` builds fresh iterators (a readcsv-backed self join reads the
  file twice).
- **Empty right stream.** A right stream producing no chunks over the
  (widened) window passes left chunks through unchanged apart from the
  `leftprefix` rename: no right columns, no `righttime`. Schemas are
  data-driven everywhere in this package, and with no right chunk there is
  no right schema to emit (or validate against).

The implementation is a single-pass two-pointer merge: a `chunkmap` over the
left stream pulls right chunks on demand — the right pointer advances per
left *row* — keeping the most recent admitted right row per key. The store's
key and row types are concrete NamedTuple types derived from the promoted
right schema (widened when a later chunk moves it, as the summarizer states
are), and the per-row merge sits behind a function barrier in the
`summarize.jl` style. `strict` and `tolerance` ride in type parameters
(`strict` as the comparison function `<` vs `<=`), so neither costs a per-row
branch.

The store is a `Dict{K, Int}` of slot numbers over a `Vector{V}` of rows,
rather than the `Dict{K, V}` it reads as — see "Representing a match" below.

## Forward join (acausal)

`futurejoin` is the forward-looking mirror of `asofjoin`: it joins each left
row to the **earliest** right row whose time is not before the left row's time
(`strict = true`: strictly after), with `missing` where none qualifies. It is
**acausal** — a row emitted at time `t` looks at right rows with time `>= t` —
so it is the one operator that deliberately breaks the causal invariant, and
it lives in the `CausalFrames.Acausal` submodule (`src/acausal.jl`), reached
only through `using CausalFrames.Acausal` and never re-exported from the top
level. Everything the top-level module exports stays causal.

It mirrors `asofjoin`'s full API (`key`, `tolerance`, `strict`, `leftprefix`,
`rightprefix`, `righttime`, curried + uncurried forms) and its streaming
machinery — a `chunkmap` over the left stream pulling right chunks on demand,
type-unstable setup once per right chunk, the per-row merge behind a function
barrier — with three inversions:

- **Match and tie-break.** The comparator `after` (`>` or `>=`) rides in a
  type parameter, and its negation is the discard test (the reverse of
  `asofjoin`'s `before`, never reused). Among right rows sharing the earliest
  qualifying time, the **first** in stream order wins (`asofjoin` keeps the
  last).
- **Context widening.** With `tolerance` the match requires
  `rtime - time <= tolerance` and the right pipeline runs over the widened
  context `[start, stop + tolerance)` — the only place times are *added*
  (`futurecontext`, mirror of `rightcontext`'s subtraction; an explicit guard
  rejects negative tolerance, which the `Context` constructor would accept).
  Without `tolerance` the right sees only `[start, stop)`, so left rows near
  `stop` may find no later right row (mirror of `asofjoin` near `start`).
- **Buffering.** Because the match is the *earliest* qualifying right row,
  the store is a `Dict{K, KeyBuffer{V}}` of per-key FIFOs (append on pull,
  logical pop-front by advancing a `head`, amortized compaction — the
  `segtree.jl` idiom) rather than one row per key. Rows are held until a left
  row consumes or outruns them, and confirming that a key has no future match
  drains the right stream, so worst-case memory is O(number of right rows) —
  the price of looking forward. `matches` stores copied row values, not buffer
  indices, so compaction never invalidates an emitted match (see "Representing
  a match" below).

The op-agnostic helpers (`normprefix`, `prefixed`, `storerowtype`,
`storekeytype`, `rowat`, `keyat`, `matchcolumn`, `convertmatches`, plus
`chunkmap`, `tokeycolumns`, `chunktypes`, `promotetypes`) are imported from the
parent module; only the small config/state-typed helpers (`checkkeys`,
`checknames`, `prefixleft!`, `assemble`) are duplicated, to carry `futurejoin`
in their error messages.

## Representing a match

Both joins accumulate one match per left row and turn them into columns when
the chunk is assembled. The obvious representation — a
`Vector{Union{Missing, V}}` over the right row type — costs a **heap
allocation per left row**, and only for the rows that actually match, which is
why it went unnoticed: it appears whenever `V` is not an `isbitstype`, and any
`String` column or any `Missing`-admitting column on the right side is enough.
Julia stores `V` inline in a `Vector{V}` but not in a `Vector{Union{Missing,
V}}`, which is a boxed-reference array unless *every* member of the union is
isbits.

So a match is a `Vector{V}` of rows plus a `Vector{Bool}` mask. Slots for
unmatched rows are left **undefined** rather than set to a sentinel, so
`matchcolumn` reads them only where the mask allows, and `convertmatches`
copies only those slots when a schema widening rebuilds a half-filled buffer
mid-chunk.

That alone fixes `futurejoin`, whose store holds *mutable* `KeyBuffer`s and so
answers a lookup with a pointer that needs no box. `asofjoin`'s store held rows
by value, so `get` had to materialise `Union{Nothing, V}` — boxing exactly the
same way, and the match array merely reused that box. Hence the `Dict{K, Int}`
of slot numbers over a `Vector{V}`: an `Int` is isbits, so the lookup is free
and the row is read back inline. Admission claims a slot with `get!`, so it
still costs one hash whether or not the key is new, and memory stays O(distinct
keys).

The rows must be *copied* into the match buffer rather than referenced by
index: `asofjoin` overwrites a key's slot when a later right row arrives, and
`futurejoin` compacts its per-key FIFOs, so an index recorded earlier in the
chunk would silently change meaning.

## Lookup join

`lookupjoin(table; key, unmatched, leftprefix, rightprefix)` joins each row to
the row of an in-memory Tables.jl table with the same key. The table has **no
time column** — one is an `ArgumentError` pointing at `asofjoin`, and a
`CausalFrame` is refused the same way — so it holds over every window: the join
is causal at every row whatever the context, and nothing needs widening.

A lookup table is not a stream, so it is never clipped to the window, and a
table with no rows still has a schema to append. (An `asofjoin` against a
constant-time stream, its predecessor, lost every row when the constant fell
outside the window.)

- **Keys.** `key` is required and must be present in both — the table's checked
  at construction, each input chunk's on arrival. Keys match with `isequal`
  and are not converted (the `KeySet` rule: an `Int` key finds a `Float64`
  one, `missing` finds `missing`). Key columns appear once, from the input row.
  A key repeated in the table is an `ArgumentError` at construction: a
  many-to-one join keeps the output row for row with the input, and a duplicate
  is far likelier a mistake than a request to multiply rows.
- **Unmatched rows.** `unmatched = :missing` (default) appends `Union{Missing, T}`
  columns. `:error` throws on the first unmatched key and `:drop` removes the
  row; both keep the table's own `T`, since every emitted row has a match. The
  element types follow from the keyword alone, never from whether a chunk
  happened to match, so the schema stays data-independent.
- **Prefixes.** `leftprefix` / `rightprefix` as for `asofjoin`, with the same
  uniqueness rule: the table's output names are checked at construction, the
  input's on every chunk.
- **Copying.** Each table column is `collect`ed when `lookupjoin` is called.
  `readtable`'s DataFrame path keeps a reference instead, but the index here is a
  derived structure a caller's mutation would silently invalidate, and lookup
  tables are small. Output columns are always new vectors.

The implementation is a stateless `chunkmap`. Construction builds a
`LookupJoin{K,C,KN,M}`, captured by the closures so each chunk's call is
statically dispatched: a `Dict{K,Int}` from key to table row (`K` the table's key
NamedTuple type), the value columns as a concrete NamedTuple, and `unmatched`
resolved to a singleton mode type. Per chunk, `lookuprows!` fills a
`Vector{Int}` of table rows, 0 for no match, behind a function barrier. Only
`Int`s are stored, so a non-isbits key costs nothing per row, the lesson of
"Representing a match". Each value column is then gathered in one typed pass
(`gathermissing`, or plain `col[rows]` in the strict modes). `:drop` slices the
chunk only when some row is unmatched.

Having no state, `lookupjoin` keeps the chunk-concatenation property over split
contexts, which neither as-of join does.

## Lead and lag

`lag` and `lead` shift every row in time by a fixed `offset`, changing only the
`:time` column and passing all others through. They are the standard
time-series lead/lag: the value observed at time `t` is the input's value at
`t - offset` for `lag` (the backward-looking, lagged view) and at `t + offset`
for `lead` (the forward-looking view). Both are stateless `chunkmap`s — no
buffering, no per-key store — and a constant shift preserves the non-decreasing
order, so no re-sort is needed.

The one subtlety is the window math. Because `load`/`stream` reject any chunk
time outside `[start, stop]`, the upstream pipeline is run over an
*oppositely-shifted* context so the shifted output lands back in `[start, stop)`:

- `lag(offset)` (`t -> t + offset`) runs upstream over `[start - offset,
  stop - offset)` and adds `offset`. Output at `t` depends only on input at
  `t - offset <= t`, so it is **causal**, lives in `src/operators.jl`, and is
  exported. `lagcontext` mirrors `asofjoin`'s `rightcontext`.
- `lead(offset)` (`t -> t - offset`) runs upstream over `[start + offset,
  stop + offset)` and subtracts `offset`. Output at `t` depends on input at
  `t + offset > t`, so it is **acausal** and lives in the `CausalFrames.Acausal`
  submodule alongside `futurejoin`, never re-exported. `leadcontext` mirrors
  `futurecontext`.

Both require the time type to support adding and subtracting the offset (numbers
and `Dates` types do), reject a negative `offset` when the pipeline runs (the
guard `rightcontext`/`futurecontext` use, since a negative shift would flip the
causality contract), and treat `offset == 0` as the identity. The shared
`shiftchunk!`/`shifttime` (broadcast add behind a function barrier) lives in
`src/operators.jl` and is imported into the submodule; `lead` shifts by
`-offset`.

## Retiming

`settime(spec)` is the general form of the constant shift: it recomputes `:time`
per row rather than moving every row by the same amount. `spec` is either a
`Symbol` — that column *becomes* `:time`, taking over the position it already
occupied, and the old `:time` disappears — or a per-row function whose result
overwrites `:time` in place. These are `readcsv`'s two `time =` modes, applied
mid-stream. Either way the result is converted to the context's time type and
the chunk is re-clipped to `[start, stop)`.

It ships as a causal/acausal pair, the same split as `lag`/`lead`:

- `settime` (`src/operators.jl`, exported) requires every row's new time to be
  at least its old one, so a row may only move later. Output at `t` then depends
  only on input at some `t' <= t`: **causal**.
- `CausalFrames.Acausal.settime` drops that requirement, so rows may move
  earlier and the output at `t` may carry what the input held later: **acausal**,
  and clipped at both ends of the window.

The acausal variant is the one exception to the submodule's export rule. Both
modules would otherwise export `settime`, and Julia makes a name exported by two
`using`d modules an error to use unqualified — so `using CausalFrames.Acausal`
would break the *causal* `settime` for everyone, including someone who only
wanted `futurejoin`. It is therefore defined in `Acausal` but left out of its
`export` list, reached as `CausalFrames.Acausal.settime`. That is strictly more
quarantined than the rule requires, which suits an escape hatch.

Three **independent** validity checks, all raised when the pipeline runs, none
implying another:

1. every row's new time is at least its old one (causal variant only);
2. the resulting column is non-decreasing within the chunk;
3. it does not step back across a chunk boundary.

(1) does not imply (3): with `spec = :x`, chunk 1 `time=[1,2], x=[5,9]` and
chunk 2 `time=[3,4], x=[6,7]` pass (1) and (2) in both chunks yet emit
`5, 9, 6, 7`. Nor does (1) imply (2) — a forward-only map can still reorder rows
within a chunk — and (2) implies neither, since mapping every row to `start` is
sorted, boundary-safe, and backward. So `settimechunk!` carries a `prevtime`
across chunks the way `clipchunk!` does.

`settimechunk!` is shared with the acausal variant exactly as `shiftchunk!` is
shared with `lead`; the two differ only in a `causal::Bool` and the operator name
used in messages, one branch per chunk and none per row. It could not reuse
`resolvetime!`: that renames onto `:time` without dropping the existing one,
which DataFrames rejects, and its error wording is welded to the file sources'
`path`/`what`. Only `maptime` is shared. The per-row causality check is an
explicit loop behind a function barrier over two concretely typed vectors —
`all(new .>= old)` would allocate a `BitVector` per chunk and lose the row index
the message wants.

**The window is not widened.** `lag`/`lead` slide their upstream window because
their shift is a constant known before any data is read (`lagcontext` /
`leadcontext`). `settime`'s shift is per-row and data-dependent, so upstream runs
over `[start, stop)` unchanged and only rows *already in the window* can be
retimed. A row whose *original* time lies outside `[start, stop)` but whose *new*
time would lie inside it is never seen: for the causal variant that means rows
before `start` that would move into the window; for the acausal variant,
additionally rows at or after `stop` that would move back into it. The same fact
costs `settime` the chunk-concatenation property — loading `[a, c)` is not the
concatenation of loading `[a, b)` and `[b, c)`, because a row retimed across `b`
is clipped away by the first evaluation and never offered to the second. Only the
`prevtime` guard crosses chunks, so streaming still equals loading. If you need
the out-of-window rows, widen the context yourself; nothing can infer how far.

`settime(:time)` is legal and leaves the values alone, but it still re-clips, so
a row sitting exactly at `stop` — which frames tolerate and `summarize` emits —
is dropped. Special-casing it into a true no-op was rejected: it would then
behave differently from `settime(r -> r.time)`, which is worse.

## Rolling windows

`addrollingcolumns(windows, ss; key, from)` is the second binary operator:
it keeps every input row and column and appends, for each named window, each
summarizer's value columns computed over that row's trailing window.
`windows` maps window names to look-backs — a NamedTuple
(`(m5 = Minute(5), h1 = Hour(1))`), a single pair, or a collection of
pairs — and each summary column is the window name prefixed onto the
summarizer's usual output name: `m5_price_sum`. A row at time `t`
summarizes the rows with time `s` satisfying `s <= t` and
`t - s <= lookback`, inclusive on both ends (the `asofjoin` tolerance
convention, not the sources' half-open one) — so under self-summarization
the row itself, and every row sharing its timestamp, is in its own window.

- **The summarized stream.** By default the summaries are computed over the
  pipeline being augmented itself, which then runs twice — pipelines are
  lazy, so each `run(ctx)` builds fresh iterators (the self-join precedent;
  a readcsv-backed pipeline reads the file twice). `from` names a different
  pipeline to summarize instead. Either way summarized rows relate to
  output rows by time (and key) only, never by row identity.
- **Context extension.** The summarized pipeline runs over the widened
  context `[minimum over windows of start - lookback, stop)`, so the first
  output row already sees a full look-back of history. Look-backs are never
  compared to *each other* (mixed `Minute`/`Hour` look-backs need not be
  comparable) — the widened starts all live in the time type, where the
  earliest is found, and a negative look-back is rejected at run time,
  generically, via `start - lookback <= start` (the `asofjoin` precedent).
  Zero look-back is legal: the window holds exactly the rows at time `t`.
- **Keys.** With `key` (a column name or collection, present in both
  inputs — validated on each side's first chunk) each row's window holds
  only the summarized rows sharing its key value, matched with `isequal`.
  `time` may not be a key.
- **Empty windows.** A window holding no rows — under a short look-back on
  a sparse stream, or a key never seen — yields the summarizers' empty
  values, so an output column's element type is the field-wise promotion of
  the summary value type with the empty value's (`Min` over an `Int` column
  gives `Union{Missing, Int}`). A summarized stream producing no chunks
  yields the empty values everywhere; with no chunk there is no schema to
  type states from, so those columns are typed from the configs alone.
- **Names.** The prefixed output names must not collide with the input's
  columns, nor with each other — distinct window names do not guarantee the
  latter (window `:a` with output `:b_x_sum` collides with window `:a_b`
  with output `:x_sum`), so uniqueness is checked over the full window ×
  output cross product, at construction time.

The implementation is a `chunkmap` over the augmented stream that pulls
summarized chunks on demand (the `asofjoin` machinery). Times are
non-decreasing, so windows slide forward monotonically: per window and key,
rows enter in stream order and expire for good, oldest first. The window
algorithm is chosen **per accumulator**, not per call (`tiers.jl`). The
expanded prototype tuple is partitioned into tiers from the realized states,
so the weakest structure in a call costs only its own summarizers:

- **Running tier — `GroupSummarizer`s.** Each window keeps per-key running
  states (a `Dict` from key to state tuple plus a live-row count). Admitted
  rows are `update!`d in, and a per-window eviction head over the shared row
  buffer `downdate!`s rows as they age out, oldest first: O(1) amortized per
  row per window. The states are the summarizers' *windowed* states
  (`freshwindowed`), which for most summarizers are their ordinary ones; `Min`,
  `Max`, `First`, `Last` and `CountDistinct` slide a windowed state of their own
  (see "Structured subtypes"). A key's group is deleted when its last row
  leaves, and retired to a pool that the next group of any key reuses zeroed,
  so an absent key *means* an empty window, and it emits the empty values
  (`Mean` over an empty window is `missing`, never `0/0`). The floating-point
  sum accumulators use compensated summation and count nonfinite terms instead
  of folding them in (see "Summarizers"), so eviction is clean: a window
  recovers exactly once a `NaN` or `±Inf` row ages out, and finite values carry
  only the compensated round-off rather than the usual sliding-sum drift.
- **Tree tier — the other `MonoidSummarizer`s.** Rows append to per-key segment
  trees (`segtree.jl`) whose nodes hold the `combine!` of their children;
  each output row binary-searches its window's start per look-back — using
  the kernel's exact membership predicate `t - s <= lookback`, never a
  rearrangement of it — and folds the window from O(log n) partial
  combinations, order-preserved for order-sensitive states. Expired rows leave
  the tree only logically (a head index) and are dropped at the next
  capacity-triggered rebuild, amortized O(1) per append; a query never
  touches a node unless its whole range is in the window, which is also
  what makes an expired absorbing leaf (`missing`, `NaN`) harmless. The trees
  own their rows, so a call mixing the tree with another tier stores its rows
  twice, once there and once in the shared buffer.
- **Re-fold tier — everything else.** Each output row folds *fresh* states
  over its window's buffered rows, from that window's eviction head: one
  `update!` per in-window row per window. The always-correct baseline, and the
  oracle the other tiers are differentially tested against.
- **No tier — stateless states.** A state with no fields (every dependent
  summarizer's) folds nothing, so it needs no per-key copy: one instance
  serves every window and reads its dependencies' values at emission.

At emission, each window takes its running group, its tree query and its
re-fold states for the row's key, splices them back into topological order by
a compile-time permutation (`mergestates`), and runs the ordinary
`summaryvalues` over the result, so a dependent reads dependencies from any
tier. Every tier holds the same rows per key, so any one of them decides
whether the window is empty. A call whose summarizers share one structure
compiles to the single-algorithm kernel: an absent tier's structure is
`nothing`, and its code disappears.

Measured over the benchmark's 100,000 rows (four per time unit, 100 keys), the
per-accumulator choice leaves single-structure calls where they were and
speeds up everything that used to fall to the weakest summarizer: an
OHLC-style `[First, Max, Min, Last, Mean, Std]` goes from 52 ms to 17 ms at a
250-unit look-back (it used to take the tree whole) and from 28 ms to 17 ms at
5 units; `[Min, Max]` from 14.4 ms (tree) to 7.6 ms (deques); and
`CountDistinct` from 218 ms (a set copied per combine) to 7.8 ms.

The partition is re-derived from the realized states whenever they are built
or widened. A widening that produces an accumulator defeating `downdate!`
(`isinvertible`) moves that accumulator from the running tier to the tree
mid-stream; the tree recovers a poisoned window once the offending row
expires, where such a running state never could. The sum family, though,
counts `NaN`, `±Inf`, and `missing` terms rather than folding them in (see
Summarizers), so those widenings stay running and recover on expiry there.
Widening only promotes, so an accumulator can demote but never return. A
widening rebuilds every tier from the live rows (the buffer, or the old trees
for a tree that already existed) — rare, O(live), and correct for every
transition; the new types force a rebuild anyway. In every tier the
type-unstable setup happens once per chunk and the per-row work sits behind
one concretely typed kernel; only the (possibly heterogeneous) look-backs
peel vararg-style, the per-window tables and value vectors being homogeneous
and indexable type-stably.

## Interval summarization

`intervalize(clock, ss; key, keyset, closelast)` is the third binary operator: a
**clock** pipeline supplies interval boundaries and the data stream is
summarized over them, the same summarizer machinery as `summarize`, dropping
the input columns. Everything after `clock` mirrors `summarize`. It relates to
`summarizecycles` as a caller-chosen grid relates to the data's own
timestamps: it is that fold with the close trigger changed from "the timestamp
changed" to "a clock boundary was crossed".

- **Boundaries.** The clock's `:time` column gives `b₀ < b₁ < … < b_K` (only
  `:time` is read; other columns are ignored, and clock order is trusted to
  the chunk protocol). Each complete interval `[bₖ, bₖ₊₁)` — inclusive of
  begin, exclusive of end, the frame convention flipped from the sources'
  `[start, stop)` only in which end is open — is summarized and emitted at its
  **end** `bₖ₊₁`. This is causal: the summary at `bₖ₊₁` folds only rows with
  time `< bₖ₊₁`. Input rows before `b₀` fall in no interval and are dropped.
- **Keyless is a regular grid.** Every complete interval emits exactly one
  row, an empty one included with the summarizers' identity/missing values
  (`emptyvalue`), so an output column's element type is the field-wise
  promotion of the summary value type with the empty value's (the
  `addrollingcolumns` empty-window rule, via the shared `promotedvaluetype`).
  A data stream producing no chunks still emits the whole empty grid, typed
  from the configs alone.
- **Keyed is sparse.** Unseen keys cannot be emitted causally, so each
  interval emits one row per key present in it, sorted by key (the
  `summarizecycles` / `closecycle!` convention); an empty interval emits
  nothing.
- **Keyed with a declared `keyset` is dense.** Every complete interval emits
  every declared key, in declared order, the keys without rows with the empty
  values — the keyless grid once per key, a data stream with no chunks
  included (see "Declared key sets").
- **The trailing partial** `[b_K, stop)` after the final boundary is emitted
  only when `closelast` (timestamped at `stop`, which frames tolerate);
  otherwise its rows are dropped. An empty clock produces no output.

The implementation is a `chunkmap` over the data stream that pulls clock
boundaries on demand. The type-unstable pull (the clock iterator and its
DataFrame column accesses are opaque, as `asofjoin`'s right stream is) is
confined to the driver: before folding a chunk it fills a concrete `Vector{T}`
of boundaries up to just past the chunk's last time, so the per-row fold
kernel takes that typed vector and stays dispatch-free (JET-checked). The
buffer carries the open interval across chunk boundaries and is trimmed to the
current interval's begin; the `SummaryFold` from the summarize transforms is
reused whole (schema promotion, state building and widening — a widening
carries the open interval's accumulated state, exactly as `summarizecycles`
does across a cycle boundary).

## Window summarization

`summarizewindows(clock, lookback, ss; key, keyset)` separates *when* a summary is
taken from *what* it covers: a clock pipeline supplies the ticks, and at each
tick `τ` the rows in the trailing window `[τ - lookback, τ)` are summarized and
emitted at `τ`, dropping the input columns. It sits between two neighbours.
`intervalize` samples at ticks but covers only the gap since the previous tick;
a look-back equal to a regular clock's spacing reproduces it at every tick
after the first. `addrollingcolumns` covers a trailing window but emits a row
per row of the stream it augments, and its keys must be present on both sides,
which a bare clock cannot supply. Per-key windows sampled at clock ticks were
the gap — fitting a model per key at every tick is the motivating case (see
"Model fitting (MLJ)").

- **Windows** are half-open: they include `τ - lookback` and exclude `τ`
  itself. That is `intervalize`'s convention, and it is what makes a summary
  emitted at `τ` fold only rows strictly before it. (`addrollingcolumns`'
  windows include their own row's time instead, because there the row *is* the
  observation being annotated.) The input runs over `[start - lookback, stop)`,
  so the first tick already sees a full window — the `rollingcontext` widening,
  with the same generic non-negativity check — and the clock over
  `[start, stop)`.
- **Keyless is a regular grid.** Every tick emits one row; an empty window
  emits the summarizers' empty values, so element types widen through the
  shared `promotedvaluetype`. No data at all still emits the whole grid, typed
  from the configs.
- **Keyed is sparse, with a vanish row.** Each tick emits one row per key with
  rows in its window, sorted by key, plus one row of empty values for every key
  that had rows at the previous tick and has none now. The key is then
  forgotten until it returns. Pure sparsity (`intervalize`'s rule) would leave
  a consumer that tracks the latest row per key — an `asofjoin`, or
  `applymodels` — holding that key's last summary forever. The vanish row is
  the one extra row that lets it see the summary go empty, without the
  ticks × keys cost of emitting every key ever seen at every tick. "Present at
  the previous tick" is decided from what was *emitted*, not from what was
  admitted, so a key whose rows arrive and age out between two ticks (a
  look-back shorter than the spacing) emits nothing at all.
- **Keyed with a declared `keyset` is dense.** Every tick emits every declared
  key, in declared order, a key with an empty window with the empty values. That
  makes the vanish row redundant, and the previous tick's keys go untracked;
  a data stream with no chunks still emits the ticks × keys grid (see
  "Declared key sets").

The mechanism is `intervalize`'s driver over `addrollingcolumns`' bookkeeping.
A `chunkmap` over the data stream pulls ticks on demand through
`intervalize.jl`'s `IntervalCursor`. Before each chunk it fills a concrete
`Vector{T}` of pending ticks up to just past the chunk's last time, so the
per-row kernel never touches the clock. For each row at time `s`, every pending
tick `τ ≤ s` is closed first — the row belongs to no window of a tick at or
before it — and then the row is admitted into a buffer of concretely typed
rows (`storerowtype`, `rowat`) with an eviction head. Closing `τ` advances the
head past the rows with `τ - time > lookback`. Flush drains the clock and
closes the remaining ticks against the buffer as it stands.

The window algorithm is chosen per accumulator, by `addrollingcolumns`' tiers
(`tiers.jl`), and re-derived whenever the states are built or widened:

- **Running**, for the `GroupSummarizer`s whose realized windowed states are
  `isinvertible`. Per-key `RunningGroup`s live in a `Dict` (keyless is the key
  `(;)`, the `rolling.jl` precedent); they are `update!`ed on admission and
  `downdate!`ed on eviction, oldest first, and a group is deleted with its last
  row (retired to a pool for reuse), so presence means rows in the window.
  Cost: O(1) amortized per row, plus a key sort per tick into a reused scratch
  vector.
- **Tree**, for the other `MonoidSummarizer`s. Rows append to per-key segment
  trees (`segtree.jl`, the `addrollingcolumns` structure), and at each tick every
  tree binary-searches its window start with the exact membership predicate
  and folds the window from O(log n) partial combinations, order-preserved for
  order-sensitive states. What differs from `addrollingcolumns` is *when* a tree
  recombines. There a window is queried after every row, so each append
  updates its O(log n) ancestors at once. Here nothing reads a tree between
  ticks, so rows append as bare leaves (`treeappend!`), and at the tick each
  tree recombines the ancestors of all its new leaves together, level by level
  (`treesync!`): O(new leaves + log n), about one `combine!` per row. An eager
  tree would pay log₂(window) combines per row (about sixteen at a 20,000-row
  window), against re-fold's one update per row per overlapping tick. A tree
  whose window empties is dropped, so presence means rows in the window, as in
  the running tier. Cost: O(1) amortized per row plus O(log n) per key per
  tick. The price is memory, as for rolling's trees: a tree holds between two
  and eight state tuples per live row, each state its own heap object, where
  re-fold holds only the rows.
- **Re-fold** otherwise. At each tick the live rows are folded into per-key
  states drawn from a `GroupTable`'s pool, emitted, and retired back —
  `closecycle!`'s protocol — at O(window) per tick. This is the path `FitModel`
  takes, and the differential-test oracle for the others.
- **No tier** for stateless (dependent) states, as in `addrollingcolumns`.

Measured over the benchmark's million rows with 1,000-unit ticks, the tiers
keep single-structure calls at parity and speed up mixed ones: the OHLC-style
set above goes from 145 ms to 122 ms keyless and 296 ms to 190 ms keyed at a
5,000-unit look-back, and from 289 ms to 119 ms keyless at a 20-unit look-back
(shorter than the tick spacing, where the old tree was rebuilt at every tick).
The one loss is keyless `[Min, Max]` alone, 48 ms to 51 ms. Per tracker and
row, the deque's `update!` on admission and `downdate!` on eviction cost 8.6–10.2
ns — a `pop!` and a `push!` on each of its two vectors, and the counters — where
the old tree zeroed and folded a leaf and recombined its share of ancestors at
the tick in 3.1–3.6 ns. The tree's own overheads (rebuilds allocating fresh
states, a window-start search per tick) win back most of that difference; the
key lookups, of the empty key, barely register. Keyed, the same set goes from
119 ms to 91 ms, since a hundred trees each rebuild and search while the
deque's cost per row is unchanged, so no heuristic chooses between them. (The buffer is
compacted at every tick's eviction, and rolling's after every row's, so it
stays near the window's size rather than growing through a whole chunk — which
is what keeps the running tier at or ahead of its old single-mode kernel.)

Presence means rows in the window in every tier, so the first tier present
(running, then tree, then re-fold) supplies the keys a tick emits, and the
others are looked up by key; each emitted row merges the tiers' states back
into topological order, as `addrollingcolumns` does. The row buffer is kept
only when the running or re-fold tier is present.

A widening that defeats `isinvertible` moves only that accumulator to the tree
mid-stream, as in `addrollingcolumns`. Every widening rebuilds the tiers from
the live rows: running groups replay the buffer, new trees replay the buffer,
and surviving trees replay their own rows; the re-fold table starts empty,
since it is filled only at ticks.

`summarizewindows` is **causal**: a row emitted at `τ` folds only rows with time
`< τ`. It is **stateful** in the usual sense, so streaming equals loading,
while split contexts do not compose: ticks realign to each context's start,
and the vanish rule depends on the previous tick.

## Declared key sets

Keyed output from `summarizecycles`, `intervalize` and `summarizewindows` is
sparse by necessity — a causal operator cannot emit a key it has not seen yet —
but the key set is often known up front, and the consumer wants a cell per
close and key, empty cells included: SQL's `keys LEFT JOIN data`. Without a
declaration that took one upstream pipeline per key, `merge`d back together.
`keyset` declares the keys and makes keyed output **dense**.

- **Every close emits every declared key, in declared order** — every cycle,
  complete interval (and the `closelast` partial), or tick. A key with no rows
  gets the summarizers' empty values, so element types widen through
  `promotedvaluetype` exactly as on the keyless grid: dense keyed output is
  that grid once per key. Declared order rather than key order leaves the order
  to the caller and skips a per-close sort; a sorted declaration reproduces the
  sparse order.
- **The declaration fixes the key type.** The key columns take their element
  types from `keyset` (a value per key for a single key column, a tuple or
  named tuple per key for several), not from the data, which is what lets a
  data stream with no chunks still emit the whole grid (`intervalize`,
  `summarizewindows`; `summarizecycles` then has no cycles and emits nothing).
  Data keys are matched by `isequal` and `hash`, so an `Int` key column matches
  a `Float64` declaration.
- **An undeclared key is an error.** A row the transform folds whose key is not
  in `keyset` throws an `ArgumentError`. Dropping it would let a typo in the
  declaration lose data silently (`filterrows` first drops on purpose), and
  emitting it sparsely would make the key type data-dependent again. Rows the
  transform discards regardless — before the first boundary, past the clock's
  last tick, in a trailing interval without `closelast` — are not checked.
- **No vanish row.** `summarizewindows` emits every declared key at every tick,
  so the row marking a key's window going empty is redundant, and the previous
  tick's keys go untracked.

`summarizecycles` and `intervalize` keep a `DenseGroups` in place of the
`GroupTable`: a state tuple per declared slot, built once, and a `folded` flag
per slot. A row pays one lookup in the `KeySet`'s `Dict{K,Int}`, what the
sparse table's lookup costs, and a close (`closedense!`) is an indexed walk over
the slots that emits them all and zeroes only the folded ones — no sort, no
table churn, no pool. `summarizewindows` keeps its tiers' structures, whose
presence-means-rows-in-the-window invariant is exactly what the dense emission
reads (`emitdense!`, a lookup per declared key per tick in the first tier
present). When that primary tier is the running or the tree tier, the
declaration is checked only where a key's group or tree is made, so a key's
later rows pay nothing for it; re-fold groups only at ticks, so when it is the
primary it checks each admitted row. The undeclared paths pass `nothing` for the key set,
and the checks dispatch away.

## Model fitting (MLJ)

Four exports put MLJ models inside pipelines:

- `FitModel`, a summarizer that fits a model to the rows it folds and emits a
  `FittedModel`;
- `applymodels`, which predicts a stream from a table of fitted models;
- `addpredictions`, the two composed over a rolling window;
- `modelreports`, which turns a table of models into a table of their fit
  reports.

**The dependency.** MLJ is reached through MLJModelInterface alone, a weak
dependency behind `ext/CausalFramesMLJModelInterfaceExt.jl` (see
`ext/CLAUDE.md`). It is the interface every model package implements, so
loading any MLJ model loads the extension, and it is small and pure Julia
(ScientificTypesBase and StatisticalTraits beyond stdlibs): measured on Julia
1.12, `using MLJModelInterface` after `using CausalFrames` — the package and
the extension together — takes about 0.01 s, against about 1 s for
CausalFrames itself. MLJBase was
rejected: machines bring Distributions, CategoricalArrays and dozens more
packages, and the model-level API (`fit`, `predict`, the data front-end
`reformat`, `save`/`restore`) needs none of them. Model *implementations*
generally do need MLJBase, for `MLJModelInterface.matrix` and friends, which is
why users load `using MLJ` — but that weight is theirs to choose. One naming
consequence: MLJ (like MLJModelInterface) exports the scientific type `Count`,
so alongside it the summarizer is written `CausalFrames.Count()`; renaming an
existing export to dodge a downstream package was not worth the breakage. As with
parquet, `src/models.jl` names no MLJ type: the extension implements five hooks
whose fallbacks live there (`ismodel`, `fitmodel`, `predictmodel`,
`savefitresult`, `restorefitresult`).

**`FitModel`** buffers the rows it folds — one concretely typed vector per
predictor and one for the response, typed from the input schema as every state
is — and fits at `value` time. A fit's cost therefore follows the host: once
per window under `summarize`, per interval under `intervalize`, per tick and
key under `summarizewindows`.
- Its output name, predictors and response are type parameters, and `value`
  builds a `NamedTuple{(N,),Tuple{FittedModel{P,M}}}` explicitly. The column
  type is thus concrete even though the fit is opaque. A `FittedModel`'s
  `fitresult` and `report` are `Any`, but they are touched once per fit and
  once per prediction group, never per row.
- It is a plain `Summarizer`, neither monoid nor group — a fit neither combines
  nor inverts — so windowed transforms re-fold for it.
- `fresh!` empties the buffers in place, keeping their capacity for the next
  window. That is safe only because `value` hands the model *copies*: a model
  that keeps its training table (a nearest-neighbour model, say) would
  otherwise see it overwritten by the next window. The copy is O(window) per
  fit, which the fit dwarfs.

**`applymodels`** is `asofjoin`'s machinery with a different assembly. The
models pipeline, narrowed to its model and key columns, is the right side of
the as-of store: `AsofJoinState`, `pullright!` and `joinsegment!` are
unchanged, and `AsofJoinConfig` carries the operator's name for error messages.
- Once a chunk's matches are known, its rows are grouped by model identity (an
  `IdDict` typed at a function barrier), and each distinct model is applied
  once, to views of its rows' predictor columns — one dynamic call per model
  per chunk.
- The results are scattered into one `Union{Missing, E}` column, with `E`
  promoted across the groups.
- The predictor names come from the `FittedModel`'s type, so a model table read
  back from disk is self-describing.
- One deliberate departure from `asofjoin`: a models stream with no rows still
  appends the prediction column, all `missing`, because the caller names that
  column rather than the data supplying it.

**`addpredictions`** is `p |> summarizewindows(clock, lookback, FitModel(...);
key)` feeding `applymodels(...; strict = false)`, and its causality argument is
the composition's. A model emitted at tick `τ` was fit on rows in
`[τ - lookback, τ)`, and a row at `t ≥ τ` uses it, so every training row is
strictly earlier than every row it predicts. That is why the windows are
half-open, and why the match need not be strict. Keyed, the vanish row of
`summarizewindows` carries a `missing` model, so a key with no rows in its
latest window is predicted `missing` rather than by a stale model; keyless, an
empty window's `missing` summary does the same. One thing causality cannot
check: the response must be observable at its row's time. A response built by
looking ahead (`Acausal.lead`) makes the whole construction look ahead, and is
the caller's explicit opt-in. The pipeline runs twice, once to fit and once to
predict, as a self-join does.

**`modelreports`** replaces the model column with each fit's report. It is the
third value `MMI.fit` returns, normalized as MLJBase's `report(mach)`
normalizes a freshly fit machine's — `MLJModelInterface.report` over the fit
report alone, which lives in MLJModelInterface itself — so the two agree
exactly: an empty report becomes `nothing`, and a model overloading `report` is
honoured. (A machine's report also merges the reports of operations run since,
which a fit-time table cannot hold.) It is row-wise and stateless, and a
`missing` cell stays `missing`. The report stays
one column. Splatting its fields into columns was rejected: a chunk's column
names must be fixed before its rows go out, and the fields are unknown until a
model has been fit, so a keyless stream opening on empty windows would have
nothing to name them from. `addcolumns` extracts fields.

**Persistence.** `FittedModel` defines `serialize`/`deserialize` on its own type
(so not piracy) that route the fitresult through `savefitresult` and
`restorefitresult` — MLJ's `save`/`restore`. Models wrapping foreign resources
therefore survive `writejls`, and a model table read back with `readjls` feeds
`applymodels` and `modelreports` directly.

## Last row

`lastrow(; key)` folds the whole window the way `summarize` does — one pass per
chunk, nothing emitted until the input is exhausted, everything produced in
`chunkmap`'s once-only `flush` — but keeps rows rather than summaries. The output
schema is the input's **exactly**: every column, in its own position, `:time`
included.

- **Keyless** emits exactly one row, the stream's last. An input with no rows
  emits nothing at all — the asymmetry with keyless `summarize`, which always
  has an identity summary to fall back on. There is no "identity row".
- **Keyed** emits one row per distinct key value, each that key's last row,
  sorted by key (the `sortedgroups` convention shared with `summarize` and
  `closecycle!`). `time` may not be a key and key columns must be unique, both
  rejected eagerly; the key columns are checked against the first chunk.

Every emitted row is **retimed to `stop`**, which is what makes sorting by key
legal: the chunk protocol demands non-decreasing times, and equal times satisfy
it whatever the row order. Frames tolerate the closed interval `[start, stop]`,
so emitting there is allowed — `summarize` already does it. The consequence is
that the original timestamp is lost; the composable recovery is
`addcolumns(r -> (; t0 = r.time))` upstream, rather than a magic extra column.

The per-key store is **join.jl's**, not a `GroupTable`: a `Dict{K,Int}` of slot
numbers over a `Vector{V}` of concretely typed rows, with `V` and `K` built by
`storerowtype`/`storekeytype` from the promoted input schema, and rows read with
`rowat`/`keyat`. The reasoning is "Representing a match" applied unchanged — a
`Dict{K,V}` can only answer `get` as `Union{Nothing,V}`, which Julia heap-boxes
whenever `V` is not isbits, and `lastrow` does a dict operation *per row*, so one
`String` column would cost one box per row. A `Dict{K,DataFrame}` of one-row
slices dodges the box only because a DataFrame is already a pointer, at the cost
of a whole DataFrames `Index` plus a one-element vector per column per key.

What `lastrow` does *not* need is the join's `found` mask: every slot a key
claims is written the same instant, so the store is never half-filled and a
widening is a plain `convert(Vector{V}, slots)` rather than `convertmatches`.
Slot numbers do not move under a widening either, so only the dict's keys are
rebuilt — `pullright!`'s pattern. Because the store is one concretely typed
vector, the flush builds the output through a `DataFrame(rows)` over it directly:
no `vcat` of per-key frames, and so no `cols = :union` question and no promotion
pass. Element types may drift chunk to chunk, as everywhere else, and the store
tracks the promotion; column *names* may not, because the row type is fixed by
them, so a chunk whose names differ in content or in order is an `ArgumentError`
rather than an opaque `convert` failure later.

The keyless path skips the store entirely — the chunk's last row *is* the last
row so far, so it costs O(ncols) per chunk and nothing per row. Routing it
through a `K = @NamedTuple{}` store would unify the code at the price of a hash
per row for no benefit, so the fold branches on the key tuple's emptiness the way
`summarize` branches on `keyed`.

`lastrow` is **causal** — a row emitted at `stop` folds only rows with time
`<= stop` — and **stateful** in the `summarize` sense: streaming equals loading
(its stream is a single frame over `[start, stop]`), while split contexts do not
compose, since each half emits its own last row per key.

## Sorting within a cycle

`sortcycles(by; rev)` stably reorders the rows of each cycle — a maximal run of
rows sharing one timestamp — and nothing else: every row keeps its time, so the
output is non-decreasing by construction and needs no order check. It is the
within-timestamp half of an SQL `ORDER BY time, ...`, the half the sources'
`sort` deliberately does not take (see "Sorting a file source"), and what makes
a keyed `Count` over `key = :time` a rank.

- **Keys.** `by` is a column name (`Symbol` or `AbstractString`), a collection of
  names compared lexicographically, or a per-row function returning the key (a
  tuple compares lexicographically). Names are normalized to a tuple of `Symbol`s
  at construction, where an empty collection or anything that is neither a name,
  a name collection nor a function is an `ArgumentError`; a named column the data
  lacks is an `ArgumentError` at run time. `:time` is not special-cased — keying
  on it reorders nothing.
- **Order.** Stable, so equal keys keep stream order, and by `isless`, so
  `missing` sorts last. `rev` reverses the whole order, `missing` first included;
  a mixed direction is a function negating a numeric key. Per-key `rev` pairs
  were left out: the function form already covers the numeric case, and a
  descending text key is rare enough not to justify a second spelling.

The mechanism is a `chunkmap` whose one piece of state is the **open cycle**: the
trailing cycle of the last chunk is held back, because the rest of it may be in
the next one. Each chunk then emits every cycle known to be complete — its prefix
before its own trailing cycle, prefixed by the held-back pieces when the chunk
closes them. A chunk whose last time equals the open time is entirely that cycle
and is appended as a piece, uncopied. The pieces are a `Vector{DataFrame}`,
concatenated once when the cycle closes, so a cycle spread over many chunks costs
O(rows) — `sortgathered`'s idiom — where re-concatenating per chunk would be
quadratic. Their column names must match, checked per piece, since a moved schema
cannot be concatenated into one cycle; element types promote on concatenation.
Flush emits the last cycle.

Nearly every chunk closes the cycle the previous one held back, so the emission
is built without concatenating the held-back tail onto the chunk, which would
copy every chunk twice. The closed cycle (the held-back pieces plus the chunk's
leading rows at that time, concatenated — a cycle's worth of rows) and the
chunk's own complete cycles are each sorted into a `SubDataFrame` view — over the
range when already in order, through the permutation otherwise — and the one or
two views are materialized together, one copy per emitted row. That copy is
unavoidable: the held-back tail means the emitted chunk is never the input chunk.

The keys are resolved once per sorted range — views of the named columns, or
`maptime`'s one pass of the function over views of the range, so the held-back
cycle is not keyed twice — and handed with a view of the time column to
`cycleperm!`, the function barrier. It walks the cycles of the range and, for
each longer than one row, first checks it for order (one pass over the index
range) and only then sorts that stretch of the permutation with Base's stable
algorithm, comparing key tuples read straight from the columns. The permutation
is allocated only when a cycle is found out of order, so an in-order stream
allocates nothing there; measured, Base's stable sort allocates no scratch either,
for four-row cycles or a single 100,000-row one. `rev` is resolved into a
concrete `Base.Order` at construction, so the comparison specializes on the
direction rather than branching on a `Bool`.

`sortcycles` is **causal** — a cycle is reordered using only its own rows, all at
its own time — and **stateful** across chunks, so streaming equals loading. It is
the one stateful operator that keeps the chunk-concatenation property over split
contexts: a split at `b` sends every row at `b` to the later half, so no cycle is
ever divided between two evaluations.

## Filling

`missing` enters a pipeline from three places — the schema union `merge`
builds, an `asofjoin` that found no right row, and a nullable parquet column —
and two transforms resolve it.

`fillmissing(specs...)` replaces `missing` with a per-column constant. It is
row-wise and stateless: a chunk is the whole context it needs. It is also the
one transform whose output element type *narrows*, to
`promote_type(nonmissingtype(T), typeof(value))` — every `missing` is
replaced, so admitting `Missing` afterwards would be a lie. Every other type
computation in the package only ever widens (`promotetypes`,
`promotedvaluetype`), so this is deliberately its own rule and not a shared
helper. A named column whose type admits no `Missing` is left untouched rather
than copied.

`forwardfill(selectors...; key, tolerance)` replaces `missing` with the
column's last non-missing value. It keeps the `Union{Missing, T}` element type,
because the rows before a column's first value — and the rows past
`tolerance` — genuinely stay missing.

The carried state is one cell per **(key, column)**, not one row per key: a
forward fill is column-independent, so `:a` may carry from row 3 while `:b`
carries from row 7, and `lastrow`'s whole-row store cannot express that. Since
a cell must be *updated* in place rather than replaced, it is a mutable struct,
and that in turn settles the store: a plain `Dict{K, NamedTuple}` of cells,
not the `Dict{K, Int}` over a slot vector [Representing a
match](#representing-a-match) argues for. The reason that store exists is that
a `Dict{K, V}` of immutable rows answers every lookup as `Union{Nothing, V}`
and boxes it whenever `V` is not isbits; a lookup of a mutable cell already
answers with a pointer, so there is nothing to box — the same reasoning that
makes `futurejoin`'s `KeyBuffer` mutable. A cell is typed at the column's
*non-missing* type, so gaining `Missing` mid-stream does not disturb it, and
the `seen` flag keeps "nothing carried yet" distinct from a column holding
`missing`, exactly as `TrackState`'s does.

A selected column whose promoted type admits no `Missing` has nothing to fill.
It gets no replacement column at all — `nothing` in the kernel's group tuple,
which folds the write away at compile time — and passes through untouched.

`tolerance` is decided per output row, against the time of the row the carried
value came from; a stale value is declined, never evicted, which is
`asofjoin`'s rule and for the same reason (a value too old for this row is not
too old for a row that shares its time). As there, `tolerance` widens the input
context to `[start - tolerance, stop)` so rows near `start` can be filled from
before the window — and, as there, without a `tolerance` there is no finite
amount to widen by, so the input sees only `[start, stop)`. The widening is the
one thing `forwardfill` must undo: `load` rejects a chunk beginning before
`start`, so the pre-window rows update the cells and are then clipped away.

Element types may move from chunk to chunk, as everywhere else, and the cells
track their promotion. The *set* of columns being filled may not: it fixes the
cell tuple's names and hence the store's type, so a chunk that changes it is an
`ArgumentError` rather than an opaque `convert` failure later — `lastrow`'s
rule, narrowed to the columns that matter here.

`forwardfill` is **causal** — the value at time `t` came from a row with time
`<= t` — and **stateful** in the streaming sense: concatenating its streamed
frames equals loading the window, while split contexts do not compose, since
the second half starts with nothing carried.

A *backward* fill would be acausal, and would have to buffer output rows until
the next non-missing value arrived — a `HeadProducer`-shaped operator in the
`Acausal` submodule, not a `chunkmap`. It is deliberately not provided.

## Summarizers

A summarization is split in two: a subtype of `Summarizer` holding only the
**configuration**, and a subtype of `SummarizerState` holding the **running
state**. The configuration is immutable and carries the column to summarize as
a *type parameter*, so the output column names it implies are known to the
compiler. The state is built from the input columns' element types, which is
what makes an output column's element type a consequence of the input schema
rather than an accident of the values it happens to hold.

The interface, extended by concrete subtypes (unexported — extend
`CausalFrames.fresh` etc.):

- `emptyvalue(s) -> NamedTuple` — the summary of no rows; also where the
  transforms read a summarizer's output column names before any data is seen;
- `fresh(s, intypes) -> SummarizerState` — a zero state, typed for input
  columns whose element types are given by `intypes` (a NamedTuple mapping
  column name to element type, mirroring `update!`'s row access: `update!`
  reads `row[column]` where `fresh` reads `intypes[column]`);
- `fresh(st) -> SummarizerState` — a zero state of the same concrete type,
  which is how the transforms get per-key-group and per-cycle states without
  re-consulting the schema;
- `fresh!(st) -> SummarizerState` — optional (defaults to `fresh(st)`); zero
  `st` in place and return it. Purely an optimization, and callers must use
  the returned value, so a state that cannot be zeroed in place simply returns
  a new one. It matters because the transforms zero a state tuple per cycle,
  per interval, per window query, and per window per row: see "Reusing state"
  below;
- `update!(st, row)` — fold one row (map-like, as for row functions) into the
  state; a summarizer reads whichever columns it needs, so multi-column
  summarizers need no special support;
- `value(st) -> NamedTuple` — the current summary; a summarizer may produce
  **several values**, and the NamedTuple's keys are the output column names
  and its value types the element types of the columns produced. Only ever
  called on a state that has folded at least one row;
- `value(st, vals) -> NamedTuple` — optional (defaults to `value(st)`);
  receives in `vals` the already-computed values of every summarizer earlier
  in topological order; see "Dependent summarizers" below;
- `widenstate(st, intypes) -> SummarizerState` — optional (defaults to `st`);
  see "Element types across chunks" below;
- `dependencies(s) -> Tuple` — optional (defaults to `()`); the summarizers
  whose values `s` reads in the two-argument `value`; see "Dependent
  summarizers" below;
- `combine!(dest, a, b)` — required of a `MonoidSummarizer`'s states (see
  "Structured subtypes" below): overwrite `dest` with the state that folding
  `a`'s rows and then `b`'s rows into a fresh state would produce. The laws:
  combination is associative, a `fresh` state is the identity on either
  side, and callers guarantee that every row folded into `a` precedes every
  row folded into `b` in stream order — which is what lets the
  order-sensitive `First`/`Last` combine. All three states are of the same
  concrete type, and `dest` may alias `a` or `b`, so implementations read
  their inputs before writing;
- `freshwindowed(s, intypes) -> SummarizerState` — optional (defaults to
  `fresh(s, intypes)`): the state a sliding window slides, which the window
  transforms `update!` as rows arrive and `downdate!` as they leave. A
  `GroupSummarizer` whose ordinary state cannot remove rows implements it
  instead. The windowed state needs `fresh`, `fresh!`, `update!`, `downdate!`,
  `value` (of the ordinary state's type) and optionally `isinvertible`; it is
  never combined or widened, since only the running window tier holds one and
  a widening rebuilds that tier from its rows;
- `downdate!(st, row)` — required of a `GroupSummarizer`'s windowed state:
  remove `row`, the **oldest** row still folded, the inverse of its `update!`.
  This is a strengthening of the plain "remove a previously folded row" a
  group inverse would be, and it is the callers' side of the contract: every
  caller evicts in the order it folded (the running tiers of
  `addrollingcolumns` and `summarizewindows` evict `buffer[head]`, and a key's
  group sees its rows in buffer order). A state may rely on it — the
  age-weighted sum knows the evicted row's weight, and the windowed trackers
  know the row is at their front — and every future caller must keep it;
- `isinvertible(st) -> Bool` — optional (defaults to `true`): whether
  `downdate!` actually inverts `update!` for the windowed state's realized
  accumulator type. The sum family keeps `NaN`, `±Inf`, and `missing` terms
  out of the running total and counts them, so they stay invertible; a state
  that folds an absorbing value past recovery returns `false`, and the window
  transforms fold that summarizer through a segment tree instead.

Output column names are deterministic, formed by suffixing the column name:
`Sum(:x)` produces `:x_sum`, `Min(:x)` produces `:x_min`, and `SumPower(:x, 2)`
and `Moment(:x, 2)` carry their exponent in the suffix to produce
`:x_sumpower_2` and `:x_moment_2`; `Count()` reads no column and produces
`:count`. `LinearRegression` is the one summarizer that departs from the
suffixing rule: it emits several columns, some of them model-level rather than
per-input, and takes an optional `name` prefixing all of them (see below).

Concrete summarizers provided, for an input column of element type `T`:

| Summarizer | Output column | Output type | Value over no rows |
|---|---|---|---|
| `Count()` | `:count` | `Int` | `0` |
| `CountDistinct(column)` | `:x_countdistinct` | `Int` | `0` |
| `Sum(column)` | `:x_sum` | `sum` of `T` | `0` |
| `SumPower(column, n)` | `:x_sumpower_2` for `n = 2` | `sum` of `T^n` | `0` |
| `Product(column)` | `:x_product` | `prod` of `T` | `1` |
| `DotProduct(a, b)` | `:a_b_dotproduct` | `sum` of `Ta * Tb` | `0` |
| `AgeWeightedSum(column)` | `:x_ageweightedsum` | `sum` of `T` | `0` |
| `Moment(column, n)` | `:x_moment_2` for `n = 2` | `sum` of `T^n` over `Int` | `missing` |
| `Mean(column)` | `:x_mean` | `sum` of `T` over `Int` | `missing` |
| `Variance(column; corrected)` | `:x_variance` | division result | `missing` |
| `Std(column; corrected)` | `:x_std` | `sqrt` of the variance | `missing` |
| `Covariance(a, b; corrected)` | `:a_b_covariance` | division result | `missing` |
| `Correlation(a, b)` | `:a_b_correlation` | division result | `missing` |
| `LinearRegression(predictors, response; intercept, name)` | `:n`, `:r2`, `:stderr`, `:intercept_beta`, `:intercept_tstat`, and `:x_beta`/`:x_tstat` per predictor | `Int` for `:n`, the division result for the rest | `0` for `:n`, `missing` for the rest |
| `Min(column)` | `:x_min` | `T` | `missing` |
| `Max(column)` | `:x_max` | `T` | `missing` |
| `First(column)` | `:x_first` | `T` | `missing` |
| `Last(column)` | `:x_last` | `T` | `missing` |
| `FitModel(model, predictors, response; name, verbosity)` | `:model`, or `name` | `FittedModel{P,M}` | `missing` |

`FitModel` is the one summarizer whose value is not a statistic but an object:
an MLJ model fit to the rows folded, buffered at the input's own types and fit
when the summary is emitted. It needs the MLJ extension, and its design is in
"Model fitting (MLJ)".

`LinearRegression` is the only summarizer emitting a block of columns: for `K`
predictors, `:n`, `:r2`, `:stderr`, then `:intercept_beta`/`:intercept_tstat`
with an intercept, then a `:p_beta`/`:p_tstat` pair per predictor in order —
`2K + 5` columns, or `2K + 3` without an intercept. An optional `name` prefixes
all of them, which is how two regressions coexist in one call; without it they
collide on `:n`, `:r2` and `:stderr` and the name-keyed deduplication rejects
them. The statistics share one element type; `:n` is the `Count` dependency
renamed, `Int` and never `missing`. No rows gives `missing` statistics and
`n = 0`; a `missing` input gives `missing` statistics and the honest count; a
rank-deficient system (collinear predictors, or `n ≤ K`, `n ≤ K + 1` with an
intercept) gives `NaN` rather than raising, as `Correlation` does for one row;
and with no residual degrees of freedom the coefficients and `:r2` are exact
while `:stderr` and the t statistics are `NaN`. Without an intercept `:r2` is
the uncentered R², as is conventional.

`Min`/`Max`/`First`/`Last` produce the input column's element type verbatim;
all four are backed by one shared state type, parameterized by the combining
function, and slide one shared windowed state (see "Structured subtypes").
`Sum` and `SumPower` produce the element type `Base.sum` would: small signed
and unsigned integers widen (`Int32` sums to `Int64`, `Bool` to `Int64`,
`UInt8` to `UInt64`), everything else keeps its type (`Float32` sums to
`Float32`). Their accumulator is built at that width up front, so the fold is
a plain `+` that cannot overflow the way accumulating in the input's own type
would. `Product` is the same story with `Base.prod`'s widening and a `*` fold.

The whole sum family (`Sum`, `SumPower`, `DotProduct`) is backed by one
shared plain state and one shared compensated state, parameterized by a
*term functor* — the same idiom as the `Min`/`Max`/`First`/`Last` state, but
for the folded quantity: the functor's type names the family and its input
columns (`ColumnTerm{:x}`, `PowerTerm{:x}`, `PairProductTerm{:a,:b}`), its
fields carry runtime config (`SumPower`'s exponent), and `update!` inlines
it statically. Every term is formed *in the accumulator's widened type* —
`SumPower` raises the widened value to the power, `DotProduct(a, b)`
multiplies widened values — so a per-row power or product cannot overflow
the way computing it in the input columns' own types would.

Carrying the exponent in a field is what makes `PowerTerm` the slow one: `^`
cannot specialize on a runtime value, so every row pays a general power where a
move or a multiply would do. `SumPower(c, 1)` and `SumPower(c, 2)` therefore
borrow the terms that hold their exponent in the *type* — `ColumnTerm{c}` and
`PairProductTerm{c,c}` — which is worth roughly 3× on the fold, and matters
well beyond `SumPower` itself since every `Variance`, `Std`, `Covariance`,
`Correlation`, and `LinearRegression` depends on the squared power sum. This is
an implementation detail: the output column keeps its own name and the
accumulator type is unchanged (`powertype(T, 1) === sumtype(T)` and
`powertype(T, 2) === dottype(T, T)`), so no schema moves. The term value is
bit-identical at `n = 1` and for integers; at `n = 2` over floats `x * x` is
the correctly rounded square, which the runtime `^` can miss by 1 ULP for
inputs whose square lands near underflow (how often depends on the CPU, since
`^` rounds its error terms differently with and without FMA, so the tests bound
the difference rather than asserting where it falls) — more accurate, but a
change. It does not
disturb what the compensated states rely on, since they classify NaN and ±Inf
*terms* and carry the sign of zero, and no nonfinite or signed-zero case
differs. `notes/sumpower-terms.md` records the measurements.

When the realized accumulator type is a fixed-precision float (a non-BigFloat
`AbstractFloat`), the sum accumulators (`Sum`, `SumPower`, `DotProduct`) switch
to a compensated state: Kahan-Babuška-Neumaier summation over the finite terms
only, with `NaN`, `+Inf`, and `-Inf` terms counted in separate `Int` fields
rather than folded in. The classified term is the folded one — the value after
`SumPower`'s power, the per-row product for `DotProduct` (so `Inf * 0.0` counts
as a `NaN` term). `value` reconstructs the IEEE result `Base.sum` would produce
(any `NaN`, or infinities of both signs, gives `NaN`; one infinity sign gives
that infinity; otherwise the compensated total), at the same declared element
type as the plain state, so nothing downstream can tell the representations
apart. Keeping nonfinites out of the running pair is what makes `downdate!` a
clean inverse for rolling windows: subtracted naively, a `NaN` absorbs and an
evicted infinity leaves `Inf - Inf = NaN` behind. BigFloat is excluded because
compensation buys nothing at arbitrary precision and a non-isbits compensated
pair would allocate on every row.

A `Missing`-admitting column gets the same treatment for `missing` that the
compensated state gives nonfinites: two flat `Optional*` states (one mirroring
the plain state, one the compensated) hold the accumulation at the *non-missing*
type and count the `missing` terms in an `Int`, folding only present terms in.
`value` returns `missing` while that count is positive and the ordinary
reconstructed total otherwise, at the declared element type `Union{Missing, A}`
— identical results to the old absorbing behaviour, but the count subtracts
away under `downdate!`, so the accumulator stays invertible and a rolling window
recovers on the running path once the missing row expires (no tree demotion).
The accumulation field is never itself `Union{Missing, …}`; only the `value`
return is. `widenstate` carries the whole representation across schema
promotions — plain→compensated (an `Int` column promoted to float), and, when
`missing` first appears, plain/compensated→`Optional*` (missings start at zero,
the existing total carried) and widening within the `Optional*` family.

`AgeWeightedSum(column)` is the sum family's one accumulator that the term
functor cannot express, because its update reads its own running total: it
folds `Σₖ k·yₖ`, where `k` is the row's age in rows (`0` for the newest). With
`Sum` and `Count` that gives the linearly weighted moving average,
`(n·Σy − Σk·y) / (n(n+1)/2)`, and least squares on the row number, as
fieldless dependents. Its state carries its own count `n` and `S₁ = Σy`
beside `S₂ = Σk·y`, since a state cannot read a sibling's: a new row adds `S₁`
to `S₂` (every row ages by one) and then joins `S₁` at weight `0`;
`downdate!`, handed the oldest row by the law above, takes back its weight
`n − 1`; and `combine!` ages `a`'s rows by `b`'s count,
`S₂ = a.S₂ + a.S₁·b.n + b.S₂`. It uses the sum family's representations: an
exact plain state for integers, and for fixed-precision floats Neumaier
compensation over `S₁` and `S₂`, with `S₁`'s counters classifying each raw
`NaN`/`±Inf` input once. `S₂`'s nonfinite terms are `S₁`'s less the newest
row's, whose weight is `0`, so a nonfinite value contributes nothing until a
later row ages it (a weight of zero is exact, not IEEE's `0·Inf = NaN`), and a
window recovers once it leaves. A `missing` input is counted as the `Optional*`
states count it, but through a type flag on the two states rather than two
more state types.

`CountDistinct` is the one summarizer that departs from both of the rules
above, and the one whose state is not O(1). It holds a `Set` of the values it
has seen, so folding `n` rows costs O(distinct) memory rather than a few fields
— the reason its docstring says so, since nothing else in the package makes a
caller think about the cardinality of a column. And a `missing` does not poison
it: `missing` is pushed into the set like any other value and the output column
is `Int`, never `Union{Missing, Int}`. The distinction is that the poisoning
rule exists because a total with an unknown term is unknowable, which a
*count of distinct values* never is — you know exactly how many distinct things
you saw, `missing` among them. `count(DISTINCT x)`'s null-skipping is a
`filterrows` upstream.

It is a group through its windowed state. The ordinary state's `Set` combines
(set union is a lawful monoid) but cannot remove a row: whether the row's value
still appears elsewhere in the window is unknowable from the set. The windowed
state counts the rows per distinct value in a `Dict{T,Int}` instead and drops a
value when its count reaches zero, so a sliding window costs O(1) per row where
the segment tree it used to take copied a set on every combine — 218 ms over
the benchmark's 100,000 rows at a 25-unit look-back. The two stay separate
because the count costs the folds that never remove a row: incrementing a count
through the public `Dict` API hashes twice per row, which measured 1.4–1.7×
slower than `push!` on a Set over a million-row fold (1.62× for 100 distinct
`Int`s, 1.39× for 100,000, 1.71× for 1,000 `Float64`s, 1.56× for 100
`String`s). A single-hash increment needs `Base`'s non-public `ht_keyindex2!`,
so `summarize`, `intervalize` and the cycle folds keep the `Set`.

`Sum`, `SumPower`, `Product`, `DotProduct`, and `CountDistinct` have an
identity element, so they summarize no rows as `0` (`Product` as `1`). The
others do not, and yield `missing` instead — reachable only
through a keyless `summarize` of an empty input, since every key group and
every cycle folds at least one row before emitting. That case is answered by
`emptyvalue` and is also the one case with no type to speak of: the chunk
protocol never yields an empty chunk, so an input with no rows carries no
schema and no state is ever built for it. `SumPower(column, 1)` produces
`:x_sumpower_1`, deliberately distinct from `Sum(column)`'s `:x_sum`, so the
two never collapse under the name-keyed deduplication described below.

### Element types across chunks

A source may hand a column a different element type from one chunk to the
next, so a column can be `Int` in one chunk and `Float64` in the next. The
summarization
transforms therefore track the promotion of every input type seen so far and
call `widenstate` when that promotion moves, rebuilding a state for the wider
type and carrying its accumulated value over. Summaries emitted before the
widening keep the narrower type, which frames already tolerate (see "Core
types"), and `DataFrame(cf)` promotes them on concatenation.

### Typing and performance

The two properties are the same mechanism. Because a state's fields are
concrete, `value(st)` returns a concretely typed NamedTuple, so the rows the
transforms collect are concretely typed, so `DataFrame` receives a known
Tables.jl schema and builds typed columns directly — there is no conversion
pass over the output.

The transforms exploit this with a **function barrier** per chunk. The
type-unstable setup — reading the schema, building or widening the states,
turning the chunk into a column table — happens once per chunk; the folding
kernels then take concretely typed arguments (a *tuple* of states, never a
`Vector{Summarizer}`; a `Dict{K,S}` of key groups with both parameters
concrete, the key names carried in a `Val`) and specialize, so the per-row
work compiles to direct field access with no dispatch or boxing. Folding a
million rows allocates on the order of kilobytes.

One consequence worth knowing: more than about 32 summarizers in a single
call — counting hidden dependencies after expansion — exceeds Julia's tuple
inference limits, and the fold degrades to dynamic dispatch. It stays
correct, just no longer specialized. The window transforms hold one tuple per
tier, which only helps, but still merge them into the whole topological tuple
at emission.

### Reusing state

A state tuple is *zeroed*, never rebuilt, wherever that happens more often than
once per chunk. States are small mutable structs, so building one per cycle or
per row is a heap allocation per summarizer on the hottest paths there are:
`summarizecycles` closes a cycle per timestamp, `intervalize` an interval per
boundary, and `addrollingcolumns` queries a window per row per window. `fresh!`
is what makes that free, and the five places that use it are `foldcycles!`,
`foldintervals!`/`flushintervals!`, `closedense!` (the declared-key cycle and
interval close, which zeroes only the slots that folded rows), the segment tree,
and the re-fold window kernel.

Three structures own reusable scratch rather than allocating it per use:

- a `SegTree` holds the two order-preserving accumulators its range queries
  fold into, so `treequery`'s result is **borrowed** — valid until that tree's
  next query, which is all its callers need, since they read it straight
  through `summaryvalues`. It also compacts its row buffers in place across
  rebuilds and, at an unchanged capacity, keeps its node vector by moving the
  live leaves' state tuples to the front by reference — nothing is re-zeroed
  or re-folded, a swapped-out tuple being zeroed only by the append that claims
  its slot — which matters because a window short enough
  to expire rows as fast as they arrive keeps the capacity at its floor and
  rebuilds every few appends;
- the re-fold window kernel threads one state tuple through its window
  recursion for a whole segment;
- a `GroupTable` — the per-key state tuples of the keyed transforms — carries a
  reused key-ordered emission buffer and a pool of retired state tuples, so
  closing a cycle neither collects a fresh vector to sort nor leaves the next
  cycle to rebuild a state tuple per key. `summarize` and `addsummarycolumns`
  never close their table and leave both buffers empty. The window
  transforms' running tier pools its retired groups the same way, which
  matters because a windowed tracker owns vectors: a key whose window empties
  and refills reuses them rather than growing new ones.

A retired or borrowed tuple is only ever handed out again after `fresh!`, and
by then `summaryvalues` has copied the values it held into the emitted row.

### Prototypes and deduplication

The summarizing transforms take one summarizer or a collection, plus an
optional `key` giving a separate summary per key value (emitted sorted by key,
or in declared order under a `keyset`; see "Declared key sets"). They treat the given summarizers as *prototypes*: they only ever
mutate `fresh` copies, one per key group (and, for `summarizecycles`, per
cycle). Before running, prototypes are **deduplicated by output-name tuple** —
identical configurations collapse to one shared instance — and the surviving
output names must be pairwise disjoint; the requested (emitted) names must
additionally be distinct from `:time` and the key columns.

### Dependent summarizers

A summarizer may compute its value from the values of other summarizers by
implementing `dependencies(s)` — a tuple of summarizer configurations — and
the two-argument `value(st, vals)`. `Moment(:x, n)`, the `n`-th raw moment,
is the built-in example: it depends on `Count()` and `SumPower(:x, n)` and
emits their quotient. `Mean`, `Variance`, `Std`, and `Covariance` are the
statistical dependents: `Mean(:x)` is `Sum(:x) / Count()`; `Variance(:x)`
combines `Count()`, `Sum(:x)`, and `SumPower(:x, 2)` by the computational
identity `(Σx² − (Σx)²/n) / (n − corrected)`; `Std(:x)` is the square root of
`Variance(:x)`; `Covariance(:x, :y)` combines `Count()`, `Sum(:x)`,
`Sum(:y)`, and the canonically ordered `DotProduct` analogously; and
`Correlation(:x, :y)` is
`Covariance(:x, :y) / (Std(:x) · Std(:y))`, clamped to `[-1, 1]`. Dependencies
may themselves be dependent — `Std` depends on `Variance`, which depends on the
raw sums, and `Correlation` depends on all three — and the topological
expansion handles that.

`Variance`, `Std`, and `Covariance` follow `Statistics`: a `corrected::Bool`
keyword (default `true`) selects the divisor `n − Int(corrected)`, so the
default is the unbiased `n − 1` estimator. `corrected` is baked into the state
type (not the output name), keeping the value fieldless and inferrable; it also
means a corrected and an uncorrected variant of the same column share an output
name and cannot be requested together in one call. `Std` clamps a
round-off-negative variance to zero before the square root, so folding never
raises a `DomainError`. `Correlation` takes no `corrected` keyword — the factor
cancels between the covariance and the standard deviations — and its result is
clamped to `[-1, 1]`, both matching `Statistics.cor`.

`LinearRegression(predictors, response)` is the largest dependent: the whole
ordinary-least-squares system, and every statistic drawn from it, is a function
of `Count()`, `SumPower(pᵢ, 2)`, `SumPower(response, 2)`, the pairwise
`DotProduct`s, and — only when there is an intercept — `Sum(pᵢ)` and
`Sum(response)`. That is what makes the sharing free: two regressions over
overlapping columns fold each cross product once, and because a squared term is
requested as `SumPower(c, 2)` rather than `DotProduct(c, c)`, and every genuine
cross product under the canonical order described below, a regression also
shares with a `Variance`, `Std`, `Correlation`, or `Covariance` the user asked
for separately, whichever way round the latter's arguments are
written. With an intercept the normal equations are centered on the column
means — the multivariate form of the `Covariance` identity, better conditioned
and one dimension smaller than carrying a column of ones — and the intercept is
recovered as `ȳ − Σᵢ βᵢ x̄ᵢ`. The system is symmetric positive semidefinite, so
it is solved by a Cholesky factorization taken with `check = false`: rank
deficiency becomes `NaN` output rather than a `PosDefException`, and the
factor's inverse supplies both the coefficients' standard errors and the
intercept's. `LinearRegression` is also the one dependent summarizer that
allocates — `K ≥ 2` builds a `K × K` workspace per emitted row — which is why
simple regression (`K = 1`) is special-cased to a closed form over scalars, the
shape that actually runs per row under `addsummarycolumns` and
`addrollingcolumns`.

### Symmetric summarizers

A summarizer whose value does not depend on the order of two column arguments
— `DotProduct(a, b)`, `Covariance(a, b)`, and every pairwise term inside
`LinearRegression` — **folds its work under one canonical argument order,
while still emitting the output column the caller asked for.** `Σab` and `Σba`
are the same number, so folding both would be duplicated per-row work; but
silently renaming the caller's column would be surprising, so
`DotProduct(:y, :x)` still produces `:y_x_dotproduct`. The canonical order is
the `isless`-sorted one, and `canonicaldot`/`canonicaldotname` in
`src/summarizers.jl` are the shared implementation.

That splits into two cases, and a new symmetric summarizer should follow
whichever fits:

- **An accumulator** — one that folds real per-row state, like `DotProduct` —
  makes its *non-canonical* form a dependent summarizer over the canonical
  one, folding nothing itself and renaming the value. `AliasState{N,D}` exists
  for exactly this: fieldless, a member of the derived-state union, emitting
  `N` from the dependency's `D`. Its `isinvertible` and `widenstate` defaults
  are already correct, since it holds no state of its own to invert or widen.
- **Something already dependent**, like `Covariance`, needs no new layer: it
  simply names the canonical form in `dependencies` and reads it back under
  the canonical name. `Covariance(:y, :x)` produces `:y_x_covariance` from the
  one `:x_y_dotproduct` accumulator.

Deduplication is by output name, so this composes: asking both ways in one
call, or asking one way beside a `LinearRegression` needing the same product,
costs one accumulator plus a free fieldless rename.

The rule is about argument *order*, and one related gap is deliberately left
open: `Covariance(:x, :x)` still depends on `DotProduct(:x, :x)` rather than
`SumPower(:x, 2)`, so it does not share with `Variance(:x)`. That is a
question of which *representation* a squared term takes, not which order its
arguments are in. `LinearRegression` resolves it in its own favour — its
diagonal terms go to `SumPower(c, 2)` — but changing `Covariance` to match
would alter an existing summarizer's dependency set for no case the regression
does not already handle.

Before running, the transforms expand the requested summarizers into the
full set to fold: each one's dependencies recursively, in topological order
by a post-order depth-first walk (dependencies may themselves be dependent;
a dependency cycle is an `ArgumentError`), deduplicated by output-name tuple
as above — so a dependency equal to a requested summarizer, or shared by two
dependents, is folded once. The dependent's state is typically fieldless:
its `update!` is a no-op and the names it reads are baked into its type
parameters, so the two-argument `value` infers. Its value's declared type
should be computed from `vals`'s *field types* (via `Base.promote_op` over
`fieldtype(typeof(vals), name)`, as `Moment` does), not `typeof` of the
runtime result — a missing-poisoned dependency would otherwise collapse the
output column's `Union{Missing, ...}` element type to `Missing`.

At emission time, values accumulate left to right over the topologically
ordered state tuple — each state's `value` sees the values of everything
before it, which is how a dependent reads its dependencies — and the
accumulated NamedTuple is then **projected down to the requested output
names**, which ride through the folding kernels in a `Val` just like the key
names. A hidden dependency is therefore folded but never emitted (and may
even share a name with a key column or, under `addsummarycolumns`, an
existing input column); requesting it alongside the dependent emits it, in
request order, from the same shared state.

### Structured subtypes

Two abstract refinements sit between `Summarizer` and the concrete types,
declaring what a summarizer's states support beyond folding:

- `MonoidSummarizer <: Summarizer` — states combine associatively over
  adjacent, stream-ordered row ranges (`combine!`), with a `fresh` state as
  the identity.
- `GroupSummarizer <: MonoidSummarizer` — updates are additionally
  invertible (`downdate!`), modulo `isinvertible`'s per-accumulator-type
  escape hatch.

`addrollingcolumns` and `summarizewindows` select their window algorithm from
this structure (see "Rolling windows" and "Window summarization"). The classification of the built-ins:

- **Groups**: `Count`, `Sum`, `SumPower`, `DotProduct`, `AgeWeightedSum` —
  subtraction is the exact inverse of addition for integer accumulators;
  float accumulators use the compensated, nonfinite-counting states (see
  above), leaving only the compensated round-off; and a `Missing`-admitting
  column counts its `missing` terms the same way, so it stays invertible
  rather than absorbing. The dependent summarizers (`Moment`, `Mean`,
  `Variance`, `Std`, `Covariance`, `Correlation`, `LinearRegression`) are
  groups too: their states are fieldless, so `combine!` and `downdate!` are
  no-ops, and the window transforms give them no tier at all — their
  effective structure is that of their dependencies.
- **Groups through a windowed state**: `Min`, `Max`, `First`, `Last`,
  `CountDistinct`. Their ordinary states have no inverse — a minimum or a set
  forgets what it would need — but combine over ordered sub-ranges (for
  `First`/`Last` *because* the ranges are ordered, which is why the law
  requires it), and fold everywhere but a sliding window at a fixed size.
  In a window they slide `freshwindowed` states instead, which the oldest-first
  law makes cheap. `Min`, `Max` and `First` share a *monotonic deque*: a new value
  first drops every value at the back it makes redundant (those `b` with
  `F(b, v)` `isequal` to `v`), so the front is always the fold of the window,
  and eviction pops the front only when the evicted row's sequence number is
  the front's. That relies on `F` selecting one of its arguments
  associatively, which `min` and `max` do under `isequal` — `NaN`, `±0.0` and
  `missing` included — as does `keepfirst` (nothing is redundant, so the deque
  is the window: O(window) memory per key). `Min` and `Max` hold between one
  value and the whole window (a monotone column). Dead front slots are
  reclaimed in place once they dominate, so a steady window allocates nothing.
  `Last` could share the deque — under `keeplast` everything is redundant, so
  it holds one value — but still pays a pop and a push on two vectors per row.
  It keeps the newest value and a count of the window's rows instead: evicting
  the oldest row changes the last value only by emptying the window, and a
  count, unlike the last row's time, tells tied rows apart. That measured
  1.1–2.2 ns per row against the deque's 5.7–8.4, and 23 ms against 38 ms for a
  keyless `summarizewindows` of `Last` over the benchmark's million rows.
  `CountDistinct` counts rows per value (see above).
- **Monoids only**: `Product` — dividing a row back out fails outright at
  zero (the total is `0` regardless of what else was folded), truncates for
  integers, and compounds round-off for floats.

A custom summarizer that declares neither still works everywhere; the window
transforms re-fold it, and only it, per window.

## Interval semantics

- **Sources** clip to the half-open interval `[start, stop)`. Adjacent
  contexts therefore tile without overlap, which is what makes chunked and
  streaming evaluation sound. The one opt-out is the `closed` keyword every
  `read*` source takes (`readtable`, `readcsv`, `readparquet`, `readjls`),
  which clips to `[start, stop]` instead; a file source defaults it to `false`,
  while a frame read back over its own context takes it by default (see
  "Tables as sources"). Closed windows over adjacent contexts overlap at the
  shared boundary, so the tiling guarantee is the caller's to give up.
- **Frames** tolerate the closed interval `[start, stop]`: intermediate
  operators may legitimately emit a row exactly at `stop` — `summarize`
  does exactly this when closing its window.

## Causality and streaming

Every operator must be **causal**: its output at time `t` may depend only on
input rows with time `≤ t` (no lookahead). Row-wise operators satisfy this
trivially; the summarization operators are causal because a summary emitted
at time `t` folds only rows with time `≤ t`.

Causality gives the *chunk-concatenation property*: for sources and row-wise
transforms, loading `[a, c)` equals concatenating the results of loading
`[a, b)` and `[b, c)`. This property is what makes chunked evaluation sound.

The deliberate exceptions are `CausalFrames.Acausal.futurejoin` (see
[Forward join (acausal)](#forward-join-acausal)), whose output at time `t`
looks at right rows with time `>= t`, `CausalFrames.Acausal.lead` (see
[Lead and lag](#lead-and-lag)), whose output at time `t` carries the input's
value from `t + offset`, and `CausalFrames.Acausal.settime` (see
[Retiming](#retiming)), whose output at `t` may carry what the input held
later. All three are quarantined in the `Acausal` submodule and never
re-exported, so opting into acausality is explicit
(`using CausalFrames.Acausal`) and everything the top-level module exports
keeps the guarantee above. `settime` goes one step further and is not exported
from the submodule either, for the name-clash reason [Retiming](#retiming)
gives.

Evaluation is streaming end to end: operators pass chunks between each other
lazily and only `load` materializes the whole window. The incremental entry
point is

```julia
stream(ctx, pipeline) # -> iterator of CausalFrames over sub-contexts
```

Chunk boundaries are **source-native**: each source picks its own batch
sizes (`clock`'s `batchsize`, `readcsv`'s `chunkbytes`); there is no time
alignment. The streamed frames' contexts tile `[start, stop)`: frame `i`
covers `[bᵢ₋₁, bᵢ)` where `b₀ = start`, `bᵢ` is the first time of chunk
`i + 1`, and the last frame's context stops at `stop`. The iterator is
single-pass and maintains one chunk of lookahead (needed to place the next
boundary).

Row-wise transforms (`filterrows`, `addcolumns`, the column projections,
`fillmissing`, `lookupjoin`, `lag`, `modelreports`) map over chunks
independently. The rest are **stateful**, carrying state across chunk
boundaries rather than restarting per chunk:

- `summarize` and `lastrow` fold the whole window and emit once, at `stop`,
  when their input is exhausted, so their stream is a single frame over
  `[start, stop]`;
- `addsummarycolumns` carries its running states, and `summarizecycles` and
  `sortcycles` the open cycle (a cycle closes when a later time arrives or the
  stream ends);
- `intervalize` and `summarizewindows` carry pending clock ticks and buffered
  rows (and, for `summarizewindows`, the previous tick's keys);
- `asofjoin`, `applymodels` (and so `addpredictions`) and `addrollingcolumns`
  carry their store or buffer of right-side rows and their position in that
  stream;
- `forwardfill` carries a value per key and column, and `head` its remaining
  row budget.

Concatenating the frames of `stream(ctx, p)` therefore always equals
`load(ctx, p)`. The chunk-concatenation property over *split contexts* does not
hold for stateful operators, with one exception: `sortcycles`, since a split at
`b` sends every row at `b` to the later half and never divides a cycle.

`settime` is the odd one out: it carries only a `prevtime` for validation, so it
is not stateful in the sense above, yet it still loses the chunk-concatenation
property, for the different reason [Retiming](#retiming) gives — a row retimed
across a split boundary is clipped by the first evaluation and never offered to
the second.

## Module layout

| File | Content |
|---|---|
| `src/CausalFrames.jl` | module, includes, exports |
| `src/context.jl` | `Context{T}` |
| `src/frame.jl` | `CausalFrame{T}`, invariants, Tables.jl interface |
| `src/chunks.jl` | internal chunk-iterator machinery (`ChunkSource`, `chunkmap`) |
| `src/pipeline.jl` | `CausalPipeline{F}`, `load`, `stream`, `scan` |
| `src/operators.jl` | sources (including the n-ary `concatenate`), the CSV sink, row-wise transforms, the causal time shift (`lag`) with the shared `shiftchunk!`, the column projections and `reordercolumns` over one shared selector vocabulary, the truncating `head` with its `HeadProducer`, and the causal retiming (`settime`) with the shared `settimechunk!` |
| `src/merge.jl` | the n-ary time-interleaving source (`Base.merge`) and its per-pipeline cursors |
| `src/parquet.jl` | the parquet operators, their docstrings, and backend selection |
| `ext/CausalFramesDuckDBExt.jl` | the DuckDB backend: the preferred reader, the fallback writer |
| `ext/CausalFramesParquet2Ext.jl` | the Parquet2 backend: the preferred writer, the fallback reader |
| `src/jls.jl` | the `Serialization`-backed persistence pair (`writejls`, `readjls`) |
| `src/table.jl` | the in-memory source (`readtable`): the generic Tables.jl path, the eagerly resolved DataFrame path, and the frame path |
| `src/summarizers.jl` | `Summarizer`/`SummarizerState` interface and the concrete summarizers |
| `src/summarize.jl` | folding kernels and the summarization transforms |
| `src/join.jl` | the as-of join transform (`asofjoin`) |
| `src/lookupjoin.jl` | the key-only join against a timeless table (`lookupjoin`) |
| `src/lastrow.jl` | the last-row-per-key transform (`lastrow`), over the join's store |
| `src/sortcycles.jl` | the within-timestamp stable sort (`sortcycles`) and its `cycleperm!` barrier |
| `src/fill.jl` | the missing-value fills: the stateful `forwardfill` and the row-wise `fillmissing` |
| `src/segtree.jl` | the monoid segment tree behind the rolling and window tree tiers |
| `src/tiers.jl` | the per-accumulator window tiers shared by `addrollingcolumns` and `summarizewindows`: partitioning, running groups, state merging |
| `src/rolling.jl` | the rolling-window summarization transform (`addrollingcolumns`) |
| `src/intervalize.jl` | the interval-summarization transform (`intervalize`) |
| `src/windows.jl` | the clock-sampled trailing-window summarization transform (`summarizewindows`) |
| `src/models.jl` | the MLJ operators (`applymodels`, `addpredictions`, `modelreports`), the extension hooks and their fallbacks, and `FittedModel`'s serializer |
| `ext/CausalFramesMLJModelInterfaceExt.jl` | the MLJ hooks for `MLJModelInterface.Model`: fit, predict, save/restore |
| `src/acausal.jl` | the `Acausal` submodule: the forward join (`futurejoin`), the acausal time shift (`lead`), and the permissive retiming (`settime`, not exported even from the submodule) |
| `src/precompile.jl` | PrecompileTools workload covering the main pipeline paths |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `stream`,
`scan`, `context`, `timetype`, `emptyframe`, `concatenate`, `clock`, `readcsv`, `writecsv`, `readparquet`,
`writeparquet`, `readjls`, `writejls`, `readtable`, `filterrows`,
`addcolumns`, `selectcolumns`, `dropcolumns`, `reordercolumns`, `Summarizer`, `MonoidSummarizer`, `GroupSummarizer`,
`SummarizerState`, `Count`, `CountDistinct`, `Sum`, `SumPower`, `AgeWeightedSum`,
`Moment`, `Product`, `DotProduct`, `Mean`, `Variance`, `Std`, `Covariance`,
`Correlation`, `LinearRegression`, `Min`, `Max`, `First`, `Last`, `FitModel`,
`FittedModel`, `applymodels`, `addpredictions`, `modelreports`, `summarize`,
`summarizecycles`, `intervalize`, `summarizewindows`, `addsummarycolumns`,
`addrollingcolumns`,
`asofjoin`, `lookupjoin`, `lag`, `settime`, `head`, `lastrow`, `sortcycles`, `forwardfill`,
`fillmissing`.

`merge` is not in that list: it is `Base.merge`, extended for `CausalPipeline`
arguments rather than exported under a name of our own, so `using CausalFrames`
leaves the dict and NamedTuple methods alone.

`CausalFrames.Acausal` and its `futurejoin`, `lead` and `settime` are
deliberately **not** in this list: the acausal operators are reached only
through `using CausalFrames.Acausal`, so acausality is always an explicit
opt-in. The submodule's `settime` is not exported from the submodule either, so
that `using CausalFrames.Acausal` cannot shadow the causal one.

Dependencies: DataFrames, CSV, Tables, LinearAlgebra, PrecompileTools, and the
`Serialization` stdlib (see "JLS I/O"); weak dependencies DuckDB and Parquet2,
each behind a package extension (see "Parquet I/O"), and MLJModelInterface,
behind the MLJ extension (see "Model fitting (MLJ)").

Package infrastructure: `test/` runs the unit tests plus an Aqua.jl quality
testset; `benchmark/benchmarks.jl` is a PkgBenchmark-compatible suite over
the hot paths; `docs/` is a Documenter.jl site built and deployed by CI.
