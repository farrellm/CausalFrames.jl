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

Public access is through:

- the Tables.jl interface — row iteration over all chunks in time order, so
  a `CausalFrame` works anywhere a Tables.jl source is accepted;
- `DataFrame(cf)` — concatenates chunks into a plain DataFrame (an explicit
  exit from the causal world, and the point where the data is copied);
- `context(cf)`, `nrow(cf)`, `names(cf)`.

### `CausalPipeline`

A lazy description of how to produce data: conceptually a function
`Context -> single-pass lazy iterator of DataFrame chunks`, with time
non-decreasing within and across chunks and empty chunks never emitted.
Nothing runs until the iterator is consumed. Two entry points evaluate a
pipeline:

- `load(ctx, pipeline) -> CausalFrame` — drains the iterator into a frame
  that wraps all the chunks, without copying; the only operation that
  forces the whole window into memory. An empty result yields a zero-row
  frame with only a `:time` column.
- `stream(ctx, pipeline) -> iterator of CausalFrames` — yields one frame per
  chunk (see "Causality and streaming" below).

## Operators

Two kinds, both compatible with the chaining operator `|>`:

- **Sources** take ordinary arguments and return a `CausalPipeline`.
- **Transforms** are curried: `filterrows(pred)` returns a
  `CausalPipeline -> CausalPipeline` function, so
  `source |> transform(args)` chains naturally.

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `clock(interval; batchsize)` | source | rows at `start, start + interval, …` while `< stop`; no other columns; generated lazily in chunks of `batchsize` rows |
| `readcsv(path; chunkbytes)` | source | CSV file with a sorted `time` column; rows clipped to `[start, stop)`; read incrementally in chunks of roughly `chunkbytes` bytes — never all at once — stopping as soon as a time `>= stop` is seen |
| `filterrows(pred)` | transform | keep rows where `pred(row)` is `true` |
| `addcolumns(f)` | transform | `f(row)` returns a `NamedTuple` of new column values for that row; may **not** contain a `time` key (this preserves the time invariant without re-validation) |
| `summarize(ss; key)` | transform | summarize the whole context into rows at time `stop`; drops input columns |
| `summarizecycles(ss; key)` | transform | summarize each cycle (maximal run of rows sharing a timestamp) independently; drops input columns |
| `addsummarycolumns(ss; key)` | transform | keep input columns, append the running summary value after each row |

Row functions (`pred`, `f`) receive a map-like row object supporting
`row.name` and `row[:name]` access (Tables.jl row semantics), including
`row.time`.

Naming follows Julia convention: lowercase, no camelCase, and no shadowing
of `Base.filter` / `Base.empty` / `Base.count` / `Base.sum`.

## Summarizers

A summarization is described by a subtype of the abstract type `Summarizer`.
An instance of a concrete subtype holds both the configuration (typically
which column(s) to summarize) and the running state. The interface, extended
by concrete subtypes (unexported — extend `CausalFrames.fresh` etc.):

- `fresh(s) -> Summarizer` — a new instance with the same configuration and
  zero state;
- `update!(s, row)` — fold one row (map-like, as for row functions) into the
  state; a summarizer reads whichever columns it needs, so multi-column
  summarizers need no special support;
- `value(s) -> NamedTuple` — the current summary; a summarizer may produce
  **several values**, and the NamedTuple's keys are the output column names.

Output column names are deterministic, formed by suffixing the column name:
`Sum(:x)` produces `:x_sum`, `Min(:x)` produces `:x_min`, and `SumPower(:x, 2)`
carries its exponent in the suffix to produce `:x_sum2`; `Count()` reads no
column and produces `:count`.

Concrete summarizers provided:

| Summarizer | Output column | Value over no rows |
|---|---|---|
| `Count()` | `:count` | `0` |
| `Sum(column)` | `:x_sum` | `0` |
| `SumPower(column, n)` | `:x_sum2` for `n = 2` | `0` |
| `Min(column)` | `:x_min` | `missing` |
| `Max(column)` | `:x_max` | `missing` |
| `First(column)` | `:x_first` | `missing` |
| `Last(column)` | `:x_last` | `missing` |

`Sum` and `SumPower` have an identity element, so they summarize no rows as
`0`. The other four do not, and yield `missing` instead — reachable only
through a keyless `summarize` of an empty input, since every key group and
every cycle folds at least one row before emitting. `SumPower(column, 1)`
produces `:x_sum1`, deliberately distinct from `Sum(column)`'s `:x_sum`, so
the two never collapse under the name-keyed deduplication described below.

The three summarization functions take one summarizer or a collection of
them, plus an optional `key` (one or more column names) to produce a separate
summary per unique key value (key groups are emitted sorted by key value).
The functions treat the given summarizers as *prototypes*: they only ever
mutate `fresh` copies, one per key group (and, for `summarizecycles`, per
cycle). Before running, prototypes are **deduplicated by output-name tuple** —
identical configurations collapse to one shared instance — and the surviving
output names must be pairwise disjoint and distinct from `:time` and the key
columns.

Planned refinements (not yet implemented):

- structured subtypes: a **monoid** subtype (mergeable state, enabling
  map-reduce evaluation) and a **group** subtype (invertible updates,
  enabling O(1) rolling windows);
- **dependent summarizers**: a summarizer will be able to declare the
  summarizers it depends on (e.g. variance depends on `Sum(:x)` and the sum
  of squares `SumPower(:x, 2)`); dependencies will be resolved through the
  same name-keyed deduplication so shared work is computed once.

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
frame over `[start, stop]`). Consequently concatenating the frames of
`stream(ctx, p)` always equals `load(ctx, p)`, even for stateful operators —
but the chunk-concatenation property over *split contexts* still does not
hold for them.

## Module layout

| File | Content |
|---|---|
| `src/CausalFrames.jl` | module, includes, exports |
| `src/context.jl` | `Context{T}` |
| `src/frame.jl` | `CausalFrame{T}`, invariants, Tables.jl interface |
| `src/chunks.jl` | internal chunk-iterator machinery (`ChunkSource`, `chunkmap`) |
| `src/pipeline.jl` | `CausalPipeline`, `load`, `stream` |
| `src/operators.jl` | sources and row-wise transforms |
| `src/summarize.jl` | `Summarizer` interface, `Count`/`Sum`, summarization transforms |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `stream`,
`context`, `emptyframe`, `clock`, `readcsv`, `filterrows`, `addcolumns`,
`Summarizer`, `Count`, `Sum`, `summarize`, `summarizecycles`,
`addsummarycolumns`.

Dependencies: DataFrames, CSV, Tables.
