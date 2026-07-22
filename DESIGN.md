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

All three apply the same O(1)-per-chunk guards (cross-chunk order, window
bounds) via the shared `checkchunk`; the O(n) within-chunk scans of the
public `CausalFrame` constructor stay the chunk protocol's responsibility.

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
| `clock(interval; batchsize)` | source | rows at `start, start + interval, …` while `< stop`; no other columns; generated lazily in chunks of `batchsize` rows |
| `readcsv(path; types, time, rename, delim, chunkbytes)` | source | CSV file, every column read as `String` unless `types` opts it into a concrete type; the time column (named `:time`, or chosen by `time` as a column name or a per-row function) must be typed and sorted; `rename` maps column names first; rows clipped to `[start, stop)`; read incrementally in chunks of roughly `chunkbytes` bytes — never all at once — stopping as soon as a time `>= stop` is seen |
| `writecsv(path; queue, ...)` | transform | transparent pass-through sink: writes each chunk to `path` as it flows by and yields it downstream unchanged (see "CSV output") |
| `filterrows(pred)` | transform | keep rows where `pred(row)` is `true` |
| `addcolumns(f)` | transform | `f(row)` returns a `NamedTuple` of new column values for that row; may **not** contain a `time` key (this preserves the time invariant without re-validation) |
| `selectcolumns(selectors...)` | transform | keep only the matching columns, in the input's own order (see "Column selectors") |
| `dropcolumns(selectors...)` | transform | keep only the non-matching columns, in the input's own order (see "Column selectors") |
| `summarize(ss; key)` | transform | summarize the whole context into rows at time `stop`; drops input columns |
| `summarizecycles(ss; key)` | transform | summarize each cycle (maximal run of rows sharing a timestamp) independently; drops input columns |
| `addsummarycolumns(ss; key)` | transform | keep input columns, append the running summary value after each row |
| `addrollingcolumns(windows, ss; key, from)` | transform | keep input columns, append each summarizer's value over each named trailing window, prefixed `"{window}_"` (see "Rolling windows") |
| `asofjoin(right; key, tolerance, strict, leftprefix, rightprefix, righttime)` | transform | left as-of join: append the most recent right row with time `<= time` (`strict`: `<`), per key; `missing` where none qualifies (see "As-of join") |

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
consumer owns the chunk it is handed, and two operators use that licence to
mutate the chunk's *column index* in place (`asofjoin`'s `prefixleft!`,
`addrollingcolumns`' `assembleempty`) — which would race the writer reading
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
left *row* — maintaining a `Dict` of the most recent admitted right row per
key. The store's key and value types are concrete NamedTuple types derived
from the promoted right schema (widened when a later chunk moves it, as the
summarizer states are), and the per-row merge sits behind a function barrier
in the `summarize.jl` style. `strict` and `tolerance` ride in type
parameters (`strict` as the comparison function `<` vs `<=`), so neither
costs a per-row branch.

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
`:count`.

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
| `Min(column)` | `:x_min` | `T` | `missing` |
| `Max(column)` | `:x_max` | `T` | `missing` |
| `First(column)` | `:x_first` | `T` | `missing` |
| `Last(column)` | `:x_last` | `T` | `missing` |

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
`Sum(:y)`, and `DotProduct(:x, :y)` analogously; and `Correlation(:x, :y)` is
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
  `missing` terms the same way, so it stays invertible rather than absorbing. The dependent summarizers (`Moment` through
  `Correlation`) are groups too: their states are fieldless, so `combine!`
  and `downdate!` are no-ops, and their effective structure is that of
  their transitive dependencies — all of which are the group accumulators
  above.
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
across augmented chunk boundaries. Consequently concatenating the frames
of `stream(ctx, p)` always equals `load(ctx, p)`, even for stateful
operators — but the chunk-concatenation property over *split contexts*
still does not hold for them.

## Module layout

| File | Content |
|---|---|
| `src/CausalFrames.jl` | module, includes, exports |
| `src/context.jl` | `Context{T}` |
| `src/frame.jl` | `CausalFrame{T}`, invariants, Tables.jl interface |
| `src/chunks.jl` | internal chunk-iterator machinery (`ChunkSource`, `chunkmap`) |
| `src/pipeline.jl` | `CausalPipeline{F}`, `load`, `stream` |
| `src/operators.jl` | sources, the CSV sink, and row-wise transforms |
| `src/summarizers.jl` | `Summarizer`/`SummarizerState` interface and the concrete summarizers |
| `src/summarize.jl` | folding kernels and the summarization transforms |
| `src/join.jl` | the as-of join transform (`asofjoin`) |
| `src/segtree.jl` | the monoid segment tree behind the rolling tree mode |
| `src/rolling.jl` | the rolling-window summarization transform (`addrollingcolumns`) |
| `src/precompile.jl` | PrecompileTools workload covering the main pipeline paths |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `stream`,
`scan`, `context`, `timetype`, `emptyframe`, `clock`, `readcsv`, `writecsv`,
`filterrows`,
`addcolumns`, `selectcolumns`, `dropcolumns`, `Summarizer`, `MonoidSummarizer`, `GroupSummarizer`,
`SummarizerState`, `Count`, `Sum`, `SumPower`,
`Moment`, `Product`, `DotProduct`, `Mean`, `Variance`, `Std`, `Covariance`,
`Correlation`, `Min`, `Max`, `First`, `Last`, `summarize`,
`summarizecycles`, `addsummarycolumns`, `addrollingcolumns`, `asofjoin`.

Dependencies: DataFrames, CSV, Tables, PrecompileTools.

Package infrastructure: `test/` runs the unit tests plus an Aqua.jl quality
testset; `benchmark/benchmarks.jl` is a PkgBenchmark-compatible suite over
the hot paths; `docs/` is a Documenter.jl site built and deployed by CI.
