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
  `Tables.partitions(cf)` yields one partition per backing chunk (as copies,
  keeping the backing opaque) for partition-aware sinks;
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
| `readcsv(path; chunkbytes)` | source | CSV file with a sorted `time` column; rows clipped to `[start, stop)`; read incrementally in chunks of roughly `chunkbytes` bytes — never all at once — stopping as soon as a time `>= stop` is seen |
| `filterrows(pred)` | transform | keep rows where `pred(row)` is `true` |
| `addcolumns(f)` | transform | `f(row)` returns a `NamedTuple` of new column values for that row; may **not** contain a `time` key (this preserves the time invariant without re-validation) |
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
summarized chunks on demand (the `asofjoin` machinery): rows with time
`<= t` are admitted into a buffer typed concretely from the promoted
summarized schema, and rows that have left every window are dropped from
its front — times are non-decreasing, so eviction is final, and the dead
prefix is compacted amortized-O(1). Summarizer states are accumulate-only —
there is no inverse of `update!` — so each output row folds *fresh* states
over its window's buffered rows, oldest to newest (`First`/`Last` depend on
the order), behind a function barrier in the `summarize.jl` style; the
per-row cost is one buffer scan plus one `update!` per in-window row per
window. The planned **group** summarizer subtype (invertible updates) is
the future O(1)-per-row refinement, and per-key buffers the refinement for
heavily keyed streams. Because states are rebuilt per row, a schema
widening mid-stream rebuilds only the state prototypes, the buffer, and the
emitted value vectors — there is no accumulated state to carry.

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
  summarizers" below.

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
would. `Product` is the same story with `Base.prod`'s widening and a `*` fold,
and `DotProduct(a, b)` sums the products `a * b`, forming each term in that
widened accumulator type so a per-row product cannot overflow either.

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

A source may infer a column's element type per chunk — `readcsv` does — so a
column can be `Int` in one chunk and `Float64` in the next. The summarization
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

Planned refinements (not yet implemented): structured subtypes — a
**monoid** subtype (mergeable state, enabling map-reduce evaluation) and a
**group** subtype (invertible updates, enabling O(1) rolling windows —
`addrollingcolumns` today re-folds each window from fresh states).

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
| `src/operators.jl` | sources and row-wise transforms |
| `src/summarizers.jl` | `Summarizer`/`SummarizerState` interface and the concrete summarizers |
| `src/summarize.jl` | folding kernels and the summarization transforms |
| `src/join.jl` | the as-of join transform (`asofjoin`) |
| `src/rolling.jl` | the rolling-window summarization transform (`addrollingcolumns`) |
| `src/precompile.jl` | PrecompileTools workload covering the main pipeline paths |

Exports: `Context`, `CausalFrame`, `CausalPipeline`, `load`, `stream`,
`context`, `timetype`, `emptyframe`, `clock`, `readcsv`, `filterrows`,
`addcolumns`, `Summarizer`, `SummarizerState`, `Count`, `Sum`, `SumPower`,
`Moment`, `Product`, `DotProduct`, `Mean`, `Variance`, `Std`, `Covariance`,
`Correlation`, `Min`, `Max`, `First`, `Last`, `summarize`,
`summarizecycles`, `addsummarycolumns`, `addrollingcolumns`, `asofjoin`.

Dependencies: DataFrames, CSV, Tables, PrecompileTools.

Package infrastructure: `test/` runs the unit tests plus an Aqua.jl quality
testset; `benchmark/benchmarks.jl` is a PkgBenchmark-compatible suite over
the hot paths; `docs/` is a Documenter.jl site built and deployed by CI.
