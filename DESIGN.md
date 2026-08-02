# CausalFrames.jl — Design

CausalFrames represents time-series tables: tabular data with a monotonically
non-decreasing `time` column. Data is described *lazily* as a pipeline;
evaluation is streaming end to end — operators pass chunks between each other
lazily. A time window (`Context`) plus a pipeline yields a materialized
`CausalFrame` via `load` (the only operation that forces the whole window
into memory) or an iterator of frames via `stream`.

```julia
using CausalFrames, Dates

p = readcsv("ticks.csv") |>
    filterrows(r -> r.price > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

frame = load(Context(DateTime(2026, 1, 1), DateTime(2026, 2, 1)), p)
```

## Core types

### `Context{T}`

A time window with fields `start::T` and `stop::T`, `start <= stop` enforced
at construction. The time type `T` is generic: anything ordered (`isless`)
works — `DateTime`, `Date`, `Int` ticks, `Float64` seconds, …

### `CausalFrame{T}`

A materialized table. **Opaque**: it hides its backing storage because a
frame is composed of one or more time-disjoint DataFrame chunks — the chunks
a pipeline streamed, wrapped without copying. Users never manipulate the
underlying DataFrames directly.

Invariants, checked at construction:

- every chunk has a `:time` column whose element type is `<: T`;
- all chunks share the same column names (element types may differ between
  chunks; `DataFrame(cf)` promotes on concatenation);
- time is non-decreasing within each chunk and across chunk boundaries;
- all times lie in the **closed** interval `[start, stop]` of the frame's
  context (see "Interval semantics" below).

The public constructors validate all of this — including an O(n) sortedness
scan — because they accept arbitrary user DataFrames. `load` and `stream`
instead construct through an internal trusted inner constructor (the
`Trusted` token): the chunk protocol they consume already guarantees the
invariants, and re-scanning each streamed chunk would tax the hot path for
nothing. They keep O(1)-per-chunk guards — cross-chunk time order and
window bounds — so a misbehaving hand-rolled `CausalPipeline` source is
still caught; within-chunk sortedness and schema equality are trusted to
the protocol (sources validate their own input, e.g. `readcsv` checks the
file's order; transforms preserve order). Any other construction site must
use the validating path.

Public access is through:

- the Tables.jl interface — a **column-access** table (`Tables.columns`
  materializes once, as a copy; row iteration is served through Tables.jl's
  row-view fallback over those columns, so consumers touching both pay one
  materialization, not two). `Tables.schema(cf)` is cheap — names from the
  first chunk, eltypes promoted across chunks without a row scan — and
  matches what `DataFrame(cf)` produces. `Tables.partitions(cf)` yields one
  partition per backing chunk (as copies, keeping the backing opaque) for
  partition-aware sinks; an empty frame yields the single zero-row frame
  `DataFrame(cf)` would, so both views agree;
- `DataFrame(cf)` — concatenates chunks into a plain DataFrame (an explicit
  exit from the causal world, and the point where the data is copied);
- `context(cf)`, `nrow(cf)`, `names(cf)`.

### `CausalPipeline`

A lazy description of how to produce data: conceptually a function
`Context -> single-pass lazy iterator of DataFrame chunks`, with time
non-decreasing within and across chunks and empty chunks never emitted.
The run function's type is a parameter (`CausalPipeline{F}`), never an
abstract `Function` field. Nothing runs until the iterator is consumed. Two entry points evaluate a
pipeline:

- `load(ctx, pipeline) -> CausalFrame` — drains the iterator into a frame
  that wraps all the chunks, without copying; the only operation that
  forces the whole window into memory. An empty result yields a zero-row
  frame with only a `:time` column.
- `stream(ctx, pipeline) -> iterator of CausalFrames` — yields one frame per
  chunk (see "Causality and streaming" below).
- `scan(ctx, pipeline) -> nothing` — drains the iterator, discarding every
  chunk. Nothing is materialized: this runs a pipeline for its side effects
  (`writecsv`) without paying for a frame that would be thrown away.

All three also have curried, context-only forms — `load(ctx)`, `stream(ctx)`
and `scan(ctx)` return a function of the pipeline — so a chain can end in its
own evaluation: `source |> transform(args) |> load(ctx)`. As with the
transforms, the two-argument form is primary and the curried one is a thin
wrapper.

`load` and `scan` apply the same O(1)-per-chunk guards (cross-chunk order,
window bounds) via the shared `checkchunk`, and `stream` applies the
equivalent ones as it places sub-context boundaries; the O(n) within-chunk
scans of the public `CausalFrame` constructor stay the chunk protocol's
responsibility.

## Operators

Two kinds, both compatible with the chaining operator `|>`:

- **Sources** take ordinary arguments and return a `CausalPipeline`.
- **Transforms** are curried: `filterrows(pred)` returns a
  `CausalPipeline -> CausalPipeline` function, so
  `source |> transform(args)` chains naturally. Each transform also has an
  uncurried, pipeline-first form `transform(p, args)` (e.g.
  `filterrows(p, pred)`), equivalent to `p |> transform(args)`, for when the
  applied form reads clearer than a chain. The curried form is primary; the
  uncurried form is a thin wrapper, so `|>` stays overhead-free.

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `concatenate(ps...)` | source | run the pipelines one after another over the same context and emit their chunks end to end; they must be passed in time order and have identical columns (see "Concatenation") |
| `merge(ps...; batchsize)` | source | run the pipelines concurrently over the same context and interleave their rows by time; columns may differ (the output is their union, `missing` where a pipeline lacks one) and ties break by argument order (see "Merging") |
| `clock(interval; batchsize)` | source | rows at `start, start + interval, …` while `< stop`; no other columns; generated lazily in chunks of `batchsize` rows |
| `readcsv(path; types, time, rename, delim, chunkbytes)` | source | CSV file, every column read as `String` unless `types` opts it into a concrete type; the time column (named `:time`, or chosen by `time` as a column name or a per-row function) must be typed and sorted; `rename` maps column names first; rows clipped to `[start, stop)`; read incrementally in chunks of roughly `chunkbytes` bytes — never all at once — stopping as soon as a time `>= stop` is seen |
| `writecsv(path; queue, ...)` | transform | transparent pass-through sink: writes each chunk to `path` as it flows by and yields it downstream unchanged (see "CSV output") |
| `readparquet(path; time, rename, backend)` | source | parquet file, read through DuckDB or Parquet2 (either backend suffices; DuckDB preferred); column types come from the file itself; the time column (named `:time`, or chosen by `time` as a column name or a per-row function) must be sorted; `rename` maps column names first; rows clipped to `[start, stop)`; read one chunk at a time — a DuckDB result chunk, or a Parquet2 row group — with the window used to skip what cannot be in it (see "Parquet I/O") |
| `writeparquet(path; queue, rowgroupsize, backend, ...)` | transform | transparent pass-through sink through Parquet2 or DuckDB (either suffices; Parquet2 preferred): buffers chunks until `rowgroupsize` rows are pending and writes them as one row group, yielding every chunk downstream unchanged; the file is valid only once finalized (see "Parquet I/O") |
| `filterrows(pred)` | transform | keep rows where `pred(row)` is `true` |
| `addcolumns(f)` | transform | `f(row)` returns a `NamedTuple` of new column values for that row; may **not** contain a `time` key (this preserves the time invariant without re-validation) |
| `selectcolumns(selectors...)` | transform | keep only the matching columns, in the input's own order (see "Column selectors") |
| `dropcolumns(selectors...)` | transform | keep only the non-matching columns, in the input's own order (see "Column selectors") |
| `summarize(ss; key)` | transform | summarize the whole context into rows at time `stop`; drops input columns |
| `summarizecycles(ss; key)` | transform | summarize each cycle (maximal run of rows sharing a timestamp) independently; drops input columns |
| `intervalize(clock, ss; key, closelast)` | transform | summarize over the intervals a clock pipeline defines (`[bₖ, bₖ₊₁)`, timestamped at `bₖ₊₁`); keyless is a regular grid, keyed is sparse (see "Interval summarization") |
| `addsummarycolumns(ss; key)` | transform | keep input columns, append the running summary value after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | keep input columns, append each summarizer's value over each named trailing window, prefixed `"{window}_"` (see "Rolling windows") |
| `asofjoin(right; key, tolerance, strict, leftprefix, rightprefix, righttime)` | transform | left as-of join: append the most recent right row with time `<= time` (`strict`: `<`), per key; `missing` where none qualifies (see "As-of join") |
| `lag(offset)` | transform | shift every row `offset` later in time (`time -> time + offset`); the value at time `t` is the input's value at `t - offset` (see "Lead and lag") |
| `settime(spec)` | transform | recompute `:time` from a column name or a per-row function; times may only move later, and the result is re-clipped to `[start, stop)` (see "Retiming") |
| `head(n)` | transform | emit the first up to `n` rows and stop pulling upstream (see "Truncation") |
| `lastrow(; key)` | transform | emit each key's last row, retimed to `stop`; keyless emits the stream's last row (see "Last row") |

Row functions (`pred`, `f`) receive a map-like row object supporting
`row.name` and `row[:name]` access (Tables.jl row semantics), including
`row.time`. The transforms iterate the concretely typed rows of a column
table behind a per-chunk function barrier — never `DataFrameRow`s, whose
column accesses are type-unstable — so a row function compiles to direct
field access, exactly like a summarizer's `update!`.

Naming follows Julia convention: lowercase, no camelCase, and no shadowing
of `Base.filter` / `Base.empty` / `Base.count` / `Base.sum` /
`Base.join`.

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
consumer owns the chunk it is handed, and three operators use that licence to
mutate the chunk's *column index* in place (`asofjoin`'s `prefixleft!`,
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
yields an empty file, never a stale one. Keyword arguments pass through to
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
stop at the first time `>= stop`), and a `rename` that leaves no unique file
column mapping to `:time` — `timesourcename` returns `nothing` rather than
guess. As with `readcsv`, sortedness is only checked in the chunks actually
read, so a violation inside a skipped row group goes unreported.

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
point when the file is the point. A stream with no rows yields a valid, empty
file. Keyword arguments pass through to `Parquet2.FileWriter`, where
`compute_statistics` defaults to `["time"]` so that files written here carry the
statistics the readers skip by; the DuckDB sink understands `compression_codec`
(mapped onto `COPY`'s `COMPRESSION`, and it records statistics of its own) and
rejects the other, Parquet2-specific options rather than silently dropping them.

## Column selectors

`selectcolumns` and `dropcolumns` project a stream onto a subset of its
columns. Both are variadic, and each selector is one of:

- a column name — a `Symbol` or an `AbstractString`;
- a `Regex`, matched against the column name with `occursin`;
- a predicate, called with the column name as a `String` (the DataFrames
  `Cols(f)` convention, and what makes `startswith("px_")` work directly);
- recursively, any collection of those.

A column matches when *any* selector matches it; `selectcolumns` keeps the
matches and `dropcolumns` keeps the rest, both in the **input's own column
order**, never the selectors'. Selecting nothing is legal (the result is a
`:time`-only stream); a projection that keeps every column passes its chunks
through untouched.

- **`:time` is implicit.** It is always kept, whatever the selectors say, and
  a `Regex` or predicate matching `"time"` is ignored rather than obeyed.
  Naming it outright in `dropcolumns` is an `ArgumentError`, raised eagerly
  at construction — the `addcolumns` rule that a row function may not return
  a `time` key, from the other direction.
- **A named column absent from the data is an `ArgumentError`**, so a typo
  fails rather than silently selecting nothing. A `Regex` or predicate
  matching nothing is not an error. Requesting zero selectors, or a selector
  that is neither a name, a pattern, a predicate, nor a collection, is an
  `ArgumentError` at construction.

The work is per column, not per row: a chunk is projected with `df[!, keep]`,
which shares the selected column vectors rather than copying them (the chunk
is owned, as in `addcolumns`). Resolving the selectors is memoized per run
against the column names it was resolved from, so a stream whose schema never
moves — the norm — runs the selectors and the validation once, on its first
chunk, while a schema that does move is re-resolved and re-validated rather
than projected through a stale column list.

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
rows enter in stream order and expire for good. The window algorithm
follows the summarizers' structure, classified over the *expanded*
prototype tuple at construction time:

- **All `GroupSummarizer`s — running mode.** Each window keeps per-key
  running states (a `Dict` from key to state tuple plus a live-row count):
  admitted rows are `update!`d in, and a per-window eviction head over the
  shared row buffer `downdate!`s rows as they age out — O(1) amortized per
  row per window. A key's group is deleted when its last row leaves, so an
  absent key *means* an empty window and emits the empty values (`Mean`
  over an empty window is `missing`, never `0/0`), exactly as the re-fold
  path's seen-flag decides it. The floating-point sum accumulators use
  compensated summation and count nonfinite terms instead of folding them
  in (see "Summarizers"), so eviction is clean: a window recovers exactly
  once a `NaN` or `±Inf` row ages out, and finite values carry only the
  compensated round-off rather than the usual sliding-sum drift.
- **All `MonoidSummarizer`s — tree mode.** Rows append to per-key segment
  trees (`segtree.jl`) whose nodes hold the `combine!` of their children;
  each output row binary-searches its window's start per look-back — using
  the kernel's exact membership predicate `t - s <= lookback`, never a
  rearrangement of it — and folds the window from O(log n) partial
  combinations, order-preserved for `First`/`Last`. Expired rows leave the
  tree only logically (a head index) and are dropped at the next
  capacity-triggered rebuild, amortized O(1) per append; a query never
  touches a node unless its whole range is in the window, which is also
  what makes an expired absorbing leaf (`missing`, `NaN`) harmless.
- **Anything less — re-fold mode.** Each output row folds *fresh* states
  over its window's buffered rows, oldest to newest, one buffer scan plus
  one `update!` per in-window row per window; the always-correct baseline,
  and the oracle the fast paths are differentially tested against.

The candidate mode is re-checked against the realized states whenever they
are built or widened: a widening that produces an accumulator defeating
`downdate!` (`isinvertible`) demotes running to tree mid-stream — the tree
recovers a poisoned window once the offending row expires, where such a
running state never could. The sum family, though, counts `NaN`, `±Inf`, and
`missing` terms rather than folding them in (see Summarizers), so those
widenings stay on the running path and recover on expiry there. Widening only
promotes, so a mode can demote but never return. A schema widening rebuilds the incremental structures from
the live rows — rare, O(live), and correct for every transition. In every
mode the type-unstable setup happens once per chunk and the per-row work
sits behind concretely-typed function barriers; only the (possibly
heterogeneous) look-backs peel vararg-style, the per-window dicts and value
vectors being homogeneous and indexable type-stably.

## Interval summarization

`intervalize(clock, ss; key, closelast)` is the third binary operator: a
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
- `downdate!(st, row)` — required of a `GroupSummarizer`'s states: remove a
  previously folded row, the inverse of `update!`;
- `isinvertible(st) -> Bool` — optional (defaults to `true`): whether
  `downdate!` actually inverts `update!` for the state's realized
  accumulator type. The sum family keeps `NaN`, `±Inf`, and `missing` terms
  out of the running total and counts them, so they stay invertible; a state
  that folds an absorbing value past recovery returns `false`.

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
| `Sum(column)` | `:x_sum` | `sum` of `T` | `0` |
| `SumPower(column, n)` | `:x_sumpower_2` for `n = 2` | `sum` of `T^n` | `0` |
| `Product(column)` | `:x_product` | `prod` of `T` | `1` |
| `DotProduct(a, b)` | `:a_b_dotproduct` | `sum` of `Ta * Tb` | `0` |
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

`LinearRegression` is the only summarizer emitting a whole block of columns, so
its contract is spelled out here. For `K` predictors it produces `2K + 5`
columns with an intercept and `2K + 3` without, in this order: `:n`, `:r2`,
`:stderr`, then `:intercept_beta` and `:intercept_tstat` when there is an
intercept, then a `Symbol(p, :_beta)`, `Symbol(p, :_tstat)` pair per predictor
`p` in the order given. An optional `name` prefixes every one of them as
`Symbol(name, :_, base)`, which is how two regressions coexist in one call —
without it they collide on `:n`, `:r2`, and `:stderr` even when their
predictors are disjoint, and the name-keyed deduplication rejects that. The
`2K + 4` statistic columns share one element type, the computation's result;
`:n` is separately `Int` and never `missing`, being the `Count` dependency
under another name. No rows gives `missing` statistics and `n = 0`; a `missing`
anywhere in a predictor or the response gives `missing` statistics and the
honest count; a rank-deficient system — collinear predictors, or `n ≤ K`
(`n ≤ K + 1` with an intercept) — gives `NaN` rather than raising, as
`Correlation` does for a single row; and with no residual degrees of freedom
left the coefficients and `:r2` are the exact fit while `:stderr` and the t
statistics are `NaN`. Without an intercept, `:r2` is the uncentered
coefficient of determination, as is conventional for a no-intercept fit.

`Min`/`Max`/`First`/`Last` produce the input column's element type verbatim;
all four are backed by one shared state type, parameterized by the combining
function.
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
the correctly rounded square, which the runtime `^` misses by 1 ULP for inputs
whose square lands near underflow — more accurate, but a change. It does not
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

`Sum`, `SumPower`, `Product`, and `DotProduct` have an identity element, so
they summarize no rows as `0` (`Product` as `1`). The others do not, and yield
`missing` instead — reachable only
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
correct, just no longer specialized.

### Reusing state

A state tuple is *zeroed*, never rebuilt, wherever that happens more often than
once per chunk. States are small mutable structs, so building one per cycle or
per row is a heap allocation per summarizer on the hottest paths there are:
`summarizecycles` closes a cycle per timestamp, `intervalize` an interval per
boundary, and `addrollingcolumns` queries a window per row per window. `fresh!`
is what makes that free, and the four places that use it are `foldcycles!`,
`foldintervals!`/`flushintervals!`, the segment tree, and the re-fold window
kernel.

Three structures own reusable scratch rather than allocating it per use:

- a `SegTree` holds the two order-preserving accumulators its range queries
  fold into, so `treequery`'s result is **borrowed** — valid until that tree's
  next query, which is all its callers need, since they read it straight
  through `summaryvalues`. It also reuses its node vector and compacts its row
  buffers in place across rebuilds, which matters because a window short enough
  to expire rows as fast as they arrive keeps the capacity at its floor and
  rebuilds every few appends;
- the re-fold window kernel threads one state tuple through its window
  recursion for a whole segment;
- a `GroupTable` — the per-key state tuples of the keyed transforms — carries a
  reused key-ordered emission buffer and a pool of retired state tuples, so
  closing a cycle neither collects a fresh vector to sort nor leaves the next
  cycle to rebuild a state tuple per key. `summarize` and `addsummarycolumns`
  never close their table and leave both buffers empty.

A retired or borrowed tuple is only ever handed out again after `fresh!`, and
by then `summaryvalues` has copied the values it held into the emitted row.

The three summarization functions take one summarizer or a collection of
them, plus an optional `key` (one or more column names) to produce a separate
summary per unique key value (key groups are emitted sorted by key value).
The functions treat the given summarizers as *prototypes*: they only ever
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

`addrollingcolumns` selects its window algorithm from this structure (see
"Rolling windows"). The classification of the built-ins:

- **Groups**: `Count`, `Sum`, `SumPower`, `DotProduct` — subtraction is the
  exact inverse of addition for integer accumulators; float accumulators
  use the compensated, nonfinite-counting states (see above), leaving only
  the compensated round-off; and a `Missing`-admitting column counts its
  `missing` terms the same way, so it stays invertible rather than absorbing. The dependent summarizers (`Moment`, `Mean`,
  `Variance`, `Std`, `Covariance`, `Correlation`, `LinearRegression`) are
  groups too: their states are fieldless, so `combine!` and `downdate!` are
  no-ops, and their effective structure is that of their transitive
  dependencies — all of which are the group accumulators above.
- **Monoids only**: `Product` — dividing a row back out fails outright at
  zero (the total is `0` regardless of what else was folded) and truncates
  for integers; `Min`/`Max`/`First`/`Last` — no inverse exists, but two
  ordered sub-ranges combine (for `First`/`Last` *because* the ranges are
  ordered, which is why the law requires it).

A custom summarizer that declares neither still works everywhere; the
rolling transform just keeps its re-fold path for any tuple containing one.

## Interval semantics

- **Sources** clip to the half-open interval `[start, stop)`. Adjacent
  contexts therefore tile without overlap, which is what makes chunked and
  streaming evaluation sound.
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

Row-wise transforms map over chunks independently. The summarization
operators are **stateful**: their state spans the whole window, carried
across chunk boundaries rather than restarting per chunk —
`addsummarycolumns` carries its running summarizers across boundaries
(preserving the chunk structure), `summarizecycles` buffers the open cycle
across boundaries (a cycle closes, causally, when a row with a later time
arrives or the stream ends), and `summarize` folds chunk by chunk and emits
once, at `stop`, when its input is exhausted (so its stream is a single
frame over `[start, stop]`). `asofjoin` is stateful too: its store of
most-recent right rows and its position in the right stream carry across
left chunk boundaries (it is causal — a row emitted at time `t` looks only
at right rows with time `<= t`, possibly from before `start` when
`tolerance` widens the right window), and so is `addrollingcolumns`, whose
buffer of summarized rows and position in the summarized stream carry
across augmented chunk boundaries. `head` and `lastrow` join that list:
`head`'s remaining-row budget spans the window (see
[Truncation](#truncation)), and `lastrow`'s per-key store does, emitting once
at `stop` exactly as `summarize` does (see [Last row](#last-row)).
Consequently concatenating the frames
of `stream(ctx, p)` always equals `load(ctx, p)`, even for stateful
operators — but the chunk-concatenation property over *split contexts*
still does not hold for them.

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
| `src/pipeline.jl` | `CausalPipeline{F}`, `load`, `stream` |
| `src/operators.jl` | sources (including the n-ary `concatenate`), the CSV sink, row-wise transforms, the causal time shift (`lag`) with the shared `shiftchunk!`, the truncating `head` with its `HeadProducer`, and the causal retiming (`settime`) with the shared `settimechunk!` |
| `src/merge.jl` | the n-ary time-interleaving source (`Base.merge`) and its per-pipeline cursors |
| `src/parquet.jl` | the parquet operators, their docstrings, and backend selection |
| `ext/CausalFramesDuckDBExt.jl` | the DuckDB backend: the preferred reader, the fallback writer |
| `ext/CausalFramesParquet2Ext.jl` | the Parquet2 backend: the preferred writer, the fallback reader |
| `src/summarizers.jl` | `Summarizer`/`SummarizerState` interface and the concrete summarizers |
| `src/summarize.jl` | folding kernels and the summarization transforms |
| `src/join.jl` | the as-of join transform (`asofjoin`) |
| `src/lastrow.jl` | the last-row-per-key transform (`lastrow`), over the join's store |
| `src/segtree.jl` | the monoid segment tree behind the rolling tree mode |
| `src/rolling.jl` | the rolling-window summarization transform (`addrollingcolumns`) |
| `src/intervalize.jl` | the interval-summarization transform (`intervalize`) |
| `src/acausal.jl` | the `Acausal` submodule: the forward join (`futurejoin`), the acausal time shift (`lead`), and the permissive retiming (`settime`, not exported even from the submodule) |
| `src/precompile.jl` | PrecompileTools workload covering the main pipeline paths |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `stream`,
`scan`, `context`, `timetype`, `emptyframe`, `concatenate`, `clock`, `readcsv`, `writecsv`, `readparquet`,
`writeparquet`, `filterrows`,
`addcolumns`, `selectcolumns`, `dropcolumns`, `Summarizer`, `MonoidSummarizer`, `GroupSummarizer`,
`SummarizerState`, `Count`, `Sum`, `SumPower`,
`Moment`, `Product`, `DotProduct`, `Mean`, `Variance`, `Std`, `Covariance`,
`Correlation`, `LinearRegression`, `Min`, `Max`, `First`, `Last`, `summarize`,
`summarizecycles`, `intervalize`, `addsummarycolumns`, `addrollingcolumns`,
`asofjoin`, `lag`, `settime`, `head`, `lastrow`.

`merge` is not in that list either: it is `Base.merge`, extended for
`CausalPipeline` arguments rather than exported under a name of our own, so
`using CausalFrames` leaves the dict and NamedTuple methods alone.

`CausalFrames.Acausal` and its `futurejoin`, `lead` and `settime` are
deliberately **not** in this list: the acausal operators are reached only
through `using CausalFrames.Acausal`, so acausality is always an explicit
opt-in. The submodule's `settime` is not exported from the submodule either, so
that `using CausalFrames.Acausal` cannot shadow the causal one.

Dependencies: DataFrames, CSV, Tables, LinearAlgebra, PrecompileTools; weak
dependencies DuckDB and Parquet2, each behind a package extension
(see "Parquet I/O").

Package infrastructure: `test/` runs the unit tests plus an Aqua.jl quality
testset; `benchmark/benchmarks.jl` is a PkgBenchmark-compatible suite over
the hot paths; `docs/` is a Documenter.jl site built and deployed by CI.
