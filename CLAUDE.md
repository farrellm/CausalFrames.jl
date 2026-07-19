# CausalFrames.jl

Julia package for time-series tables: DataFrames with a monotonically
non-decreasing `time` column, built lazily from `|>`-composable pipelines.

**DESIGN.md is the source of truth for the design and must be kept in sync
with any API or semantics change.**

## Commands

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Add a dependency: `julia --project -e 'using Pkg; Pkg.add("Name")'`
  (never hand-edit UUIDs; test-only deps also need an `[extras]` entry)
- CI tests Julia 1.10 (minimum supported), 1.12, and pre-release — don't
  use post-1.10 language/stdlib features
- The default branch is `master`, not `main` — target PRs there

## Architecture

- `src/context.jl` — `Context{T}`: time window, generic ordered time type
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-disjoint DataFrame chunks; invariants checked in the inner
  constructor; Tables.jl interface; `DataFrame(frame)` is the copy point
- `src/chunks.jl` — internal chunk protocol: `ChunkSource` and `chunkmap`,
  single-pass lazy iterators of non-empty DataFrame chunks
- `src/pipeline.jl` — `CausalPipeline{F}` (lazy `Context -> iterator of
  DataFrame chunks`; the run function's type is a parameter, not an abstract
  `Function` field), `load` (drains into one frame without copying — the
  only operation that materializes the whole window), `stream` (one frame
  per chunk)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`; row functions run over concretely typed column
  table rows behind a per-chunk function barrier, never `DataFrameRow`s
- `src/summarizers.jl` — `Summarizer` (immutable config, column name in a
  type parameter) and `SummarizerState` (running state, typed from the input
  schema) plus their interface (`emptyvalue`, `fresh`, `update!`, `value`,
  `widenstate`, `dependencies`, `combine!`, `downdate!`, `isinvertible` —
  unexported), the structured subtypes `MonoidSummarizer` (associative
  `combine!` over stream-ordered ranges, fresh state as identity) and
  `GroupSummarizer <: MonoidSummarizer` (invertible via `downdate!`), and
  the concrete summarizers (`Min`/`Max`/`First`/`Last` share one state
  type, parameterized by the combiner — monoids only, as is `Product`;
  the accumulators and all dependent summarizers are groups; `Moment` is
  the dependent-summarizer example, reading its dependencies' values
  through the two-argument `value(st, vals)`; float sum accumulators use
  the shared `Compensated` pair — Neumaier summation over finite terms,
  NaN/±Inf counted separately, IEEE result reconstructed in `value`)
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
- `src/precompile.jl` — PrecompileTools workload over the main paths

## Invariants and conventions

- Sources clip to the half-open interval `[start, stop)`; frames tolerate
  the closed interval `[start, stop]` (intermediate ops may emit at `stop`).
- Every operator must be *causal*: output at time `t` depends only on input
  rows with time `<= t`. This guarantees the chunk-concatenation property
  that streaming will rely on.
- Never expose the backing DataFrames of a `CausalFrame`; `DataFrame(frame)`
  copies. The internal chunk protocol yields only non-empty chunks; `load`
  of an empty stream gives a zero-row frame with only `:time`.
- Naming is Julian: lowercase, no camelCase, no shadowing of Base functions
  (`filterrows` not `filter`, `emptyframe` not `empty`).
- A summarizer's output column takes its element type from the input column
  (`Sum`/`SumPower` widen as `Base.sum` does). The summarization transforms
  keep their per-row folding behind a function barrier taking concretely typed
  arguments — don't reintroduce `Vector{Summarizer}`, `::Any` state fields, or
  `Dict{Any,...}` group tables on those paths.
