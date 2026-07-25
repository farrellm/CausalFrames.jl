# CausalFrames.jl — module map

DESIGN.md's "Module layout" table is the canonical index; this file records the
design rationale and performance constraints behind each module.

- `src/context.jl` / `src/chunks.jl` — `Context{T}`, the `[start, stop)`
  evaluation window, and the internal chunk protocol (`ChunkSource`,
  `chunkmap`): a single-pass lazy iterator of non-empty DataFrame chunks,
  consumers taking ownership of what they're yielded; empty chunks are
  filtered out here so all downstream code may assume a chunk has rows
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-disjoint DataFrame chunks; invariants checked in the public inner
  constructor, while `load`/`stream` build through a `Trusted`-token
  constructor (O(n) scans skipped; O(1) cross-chunk/bounds guards kept);
  column-access Tables.jl interface with a cheap `Tables.schema`;
  `DataFrame(frame)` is the copy point
- `src/pipeline.jl` — `CausalPipeline{F}` (lazy `Context -> iterator of
  DataFrame chunks`; the run function's type is a parameter, not an abstract
  `Function` field), `load` (drains into one frame without copying — the
  only operation that materializes the whole window), `stream` (one frame
  per chunk)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`; row functions run over concretely typed column
  table rows behind a per-chunk function barrier, never `DataFrameRow`s;
  `clipchunk!` (rename, resolve `:time`, sortedness, clip, convert) and the
  `ChunkSink` background writer are shared with the parquet operators.
  `readcsv` never infers types — every column is `String` unless `types`
  opts it into a concrete one (`notes/readcsv-stringtype.md` records why
  CSV.jl's own `stringtype` default stays out). `lag` shifts times via the
  shared `shiftchunk!` and widens the context in `lagcontext`, the mirror
  of `Acausal.lead`
- `src/merge.jl` — `Base.merge(ps::CausalPipeline...)`, the n-ary
  time-interleaving source (extends Base rather than shadowing it; no
  zero-arg form, which would capture `merge()`): one `MergeCursor` per
  pipeline buffering a chunk, `pickwinner` choosing by `(time, argument
  index)`, and piece-at-a-time claiming (`searchsorted*` runs, no data moved
  until `batchsize` rows are pending, then one allocation per output column
  and one copy per row; a lone whole-chunk piece is adopted, or passed
  through untouched when it already has the union schema)
- `src/parquet.jl` — `readparquet`/`writeparquet`: the API, the docstrings
  and backend selection (`resolvebackend`, `parquetproducer`, `parquetsink`,
  `backendloaded`), none of which name a backend. DuckDB and Parquet2 are
  **weak** deps behind package extensions and **either one serves both
  directions**: reading prefers DuckDB (filtered streaming query) and falls
  back to Parquet2 (row groups skipped by their own statistics); writing
  prefers Parquet2 (row groups written as the stream flows by) and falls
  back to DuckDB (stream staged in a temp table, one `COPY` at the end).
  Skipping is always an optimization — chunks are clipped again on arrival,
  so results never depend on it. `backend = :duckdb`/`:parquet2` forces the
  choice, which is how the tests cover all four combinations in one process;
  working on parquet means loading `DuckDB`/`Parquet2` in the session first
- `src/summarizers.jl` — `Summarizer` (immutable config, output column name in
  a type parameter) and `SummarizerState` (running state, typed from the input
  schema), plus their unexported interface: `emptyvalue`, `fresh`, `update!`,
  `value`, `widenstate`, `dependencies`, `combine!`, `downdate!`,
  `isinvertible`. Within that:
  - the structured subtypes are `MonoidSummarizer` (states combine
    associatively over stream-ordered ranges via `combine!`, fresh state as
    the identity) and `GroupSummarizer <: MonoidSummarizer` (also invertible,
    via `downdate!`). This split is what `rolling.jl` dispatches its window
    algorithm on, so which one a new summarizer claims is a performance
    decision, not a taxonomy one
  - `Min`/`Max`/`First`/`Last` share one state type (`TrackState`,
    parameterized by the combiner) and are monoids only, as is `Product`; the
    accumulators and every dependent summarizer are groups
  - the dependent summarizers (`Moment`, `Mean`, `Variance`, `Std`,
    `Covariance`, `Correlation`) carry no state of their own — their state
    structs are empty. They declare `dependencies` and read those values back
    through the two-argument `value(st, vals)`
  - the sum family `Sum`/`SumPower`/`DotProduct` shares one plain and one
    compensated state over a term functor (`ColumnTerm`/`PowerTerm`/
    `PairProductTerm`, terms formed at accumulator width). The `Compensated`
    pair runs Neumaier summation over the finite terms and counts NaN/±Inf
    separately, reconstructing the IEEE result in `value` — that separation is
    what lets a rolling window evict a nonfinite row cleanly and stay on the
    running path. `BigFloat` is excluded on purpose: compensation buys nothing
    at arbitrary precision, and a non-isbits `Compensated` would allocate per
    row
  - a `Missing`-admitting accumulator type gets the flat `Optional*` counting
    states over the non-missing type, counting `missing` terms exactly as the
    compensated states count nonfinites, so the accumulator stays invertible —
    no `Union{Missing,_}` accumulation field, only in `value`'s return
- `src/summarize.jl` — the folding kernels and the transforms `summarize`,
  `summarizecycles`, `addsummarycolumns`; `prototypes` expands dependencies
  topologically and returns the requested output names, which ride through
  the kernels in a `Val` to project hidden dependencies out of the output;
  per-run mutable state lives in the `SummaryFold` struct, never in
  reassigned closure captures (those get boxed)
- `src/join.jl` — `asofjoin`, the binary as-of join transform: a chunkmap
  over the left stream pulls right chunks on demand (two-pointer merge, per
  left row) into a concretely typed per-key store; `tolerance` widens the
  right context by `start - tolerance` (the one place times are subtracted)
- `src/segtree.jl` — the monoid segment tree behind the rolling tree mode:
  implicit array tree of `combine!`d partial state tuples, append-only rows,
  logical front expiry (`head`), amortized rebuilds, order-preserving
  two-accumulator range queries (`treepush!`, `treequery`, `windowstart`)
- `src/rolling.jl` — `addrollingcolumns` picks its window algorithm from the
  expanded prototype tuple's structure: all-group → per-key running states
  with per-window eviction heads, O(1)/row (`rollsegmentrunning!`);
  all-monoid → per-key segment trees, O(log n)/row (`rollsegmenttree!`);
  otherwise the re-fold baseline (`rollsegment!`, the differential-test
  oracle). Running demotes to tree when widening lets `missing` into an
  accumulator (`isinvertible`); float sums stay running because the
  compensated states evict NaN/±Inf rows cleanly; widening rebuilds
  structures from live rows
- `src/intervalize.jl` — `intervalize`, the third binary transform: summarize
  over the intervals a `clock` pipeline defines (`[bₖ, bₖ₊₁)`, timestamped at
  `bₖ₊₁`). The `summarizecycles` fold with the close trigger driven by clock
  boundaries; keyless emits a regular grid (empty intervals get `emptyvalue`s,
  output types promoted via the shared `promotedvaluetype`), keyed is sparse
  (present keys only, via `closecycle!`). The `chunkmap` over the data stream
  pulls clock boundaries into a concrete `Vector{T}` per chunk (the pull is the
  only dynamism; the per-row kernel stays dispatch-free), reusing `SummaryFold`
  whole; `closelast` closes the trailing partial at `stop`
- `src/acausal.jl` — the `Acausal` submodule (`futurejoin`, `lead`), reached
  only through `using CausalFrames.Acausal` and never re-exported, so
  acausality is always an explicit opt-in. `futurejoin` mirrors `asofjoin`'s
  streaming machinery with the match direction, the tie-break, and the context
  widening (`stop + tolerance`) all inverted; because it matches the *earliest*
  qualifying right row it must buffer right rows per key until a left row
  consumes or outruns them, and proving a key has no future match drains the
  right stream — worst case O(right rows), against `asofjoin`'s O(keys)
- `src/precompile.jl` — the PrecompileTools workload over the main pipeline
  paths; every new operator adds a path here
