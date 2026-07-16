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
- `src/pipeline.jl` — `CausalPipeline` (lazy `Context -> iterator of
  DataFrame chunks`), `load` (drains into one frame without copying — the
  only operation that materializes the whole window), `stream` (one frame
  per chunk)
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`
- `src/summarize.jl` — `Summarizer` abstract type and interface (`fresh`,
  `update!`, `value` — unexported), `Count`/`Sum`, and the summarization
  transforms `summarize`, `summarizecycles`, `addsummarycolumns`

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
