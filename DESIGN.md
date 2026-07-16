# CausalFrames.jl — Design

CausalFrames represents time-series tables: tabular data with a monotonically
non-decreasing `time` column. Data is described *lazily* as a pipeline; a
time window (`Context`) plus a pipeline yields a materialized `CausalFrame`
via `load`.

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
frame may be composed of multiple time-disjoint DataFrame chunks (the basis
for streaming). Users never manipulate the underlying DataFrames directly.

Invariants, checked at construction:

- every chunk has a `:time` column whose element type is `T`;
- all chunks share the same schema;
- time is non-decreasing within each chunk and across chunk boundaries;
- all times lie in the **closed** interval `[start, stop]` of the frame's
  context (see "Interval semantics" below).

Public access is through:

- the Tables.jl interface — row iteration over all chunks in time order, so
  a `CausalFrame` works anywhere a Tables.jl source is accepted;
- `DataFrame(cf)` — concatenates chunks into a plain DataFrame (an explicit
  exit from the causal world);
- `context(cf)`, `nrow(cf)`, `names(cf)`.

### `CausalPipeline`

A lazy description of how to produce a frame: conceptually a function
`Context -> CausalFrame`. `load(ctx, pipeline)` runs it.

## Operators

Two kinds, both compatible with the chaining operator `|>`:

- **Sources** take ordinary arguments and return a `CausalPipeline`.
- **Transforms** are curried: `filterrows(pred)` returns a
  `CausalPipeline -> CausalPipeline` function, so
  `source |> transform(args)` chains naturally.

| Operator | Kind | Semantics |
|---|---|---|
| `emptyframe()` | source | zero rows, just a `:time` column |
| `clock(interval)` | source | rows at `start, start + interval, …` while `< stop`; no other columns |
| `readcsv(path)` | source | CSV file with a sorted `time` column; rows clipped to `[start, stop)` |
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
`Sum(:x)` produces `:x_sum`; `Count()` reads no column and produces `:count`.

Concrete summarizers provided: `Count()` and `Sum(column)`. The sum of no
rows is `0`.

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
  summarizers it depends on (e.g. variance depends on sum and sum of
  squares); dependencies will be resolved through the same name-keyed
  deduplication so shared work is computed once.

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
transforms, loading `[a, c)` equals concatenating the chunks of loading
`[a, b)` and `[b, c)`. This property is why `CausalFrame` is chunk-based and
is the foundation for future incremental loading.

The summarization operators are the first **stateful** operators, and the
chunk-concatenation property does not hold for them (their state spans the
whole window — e.g. `addsummarycolumns` carries its running summarizers
across chunk boundaries, and a `summarizecycles` cycle may span a chunk
boundary). This is expected: under streaming, stateful operators will carry
their state across chunk boundaries rather than restart per chunk.

v1 implements only eager `load`. The planned streaming entry point is

```julia
stream(ctx, pipeline; chunk) # -> iterator of CausalFrames over sub-contexts
```

where stateful (aggregating) operators will carry their state across chunk
boundaries. Not yet implemented.

## Module layout

| File | Content |
|---|---|
| `src/CausalFrames.jl` | module, includes, exports |
| `src/context.jl` | `Context{T}` |
| `src/frame.jl` | `CausalFrame{T}`, invariants, Tables.jl interface |
| `src/pipeline.jl` | `CausalPipeline`, `load` |
| `src/operators.jl` | sources and row-wise transforms |
| `src/summarize.jl` | `Summarizer` interface, `Count`/`Sum`, summarization transforms |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `context`,
`emptyframe`, `clock`, `readcsv`, `filterrows`, `addcolumns`, `Summarizer`,
`Count`, `Sum`, `summarize`, `summarizecycles`, `addsummarycolumns`.

Dependencies: DataFrames, CSV, Tables.
