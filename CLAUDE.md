# CausalFrames.jl

Julia package for time-series tables: DataFrames with a monotonically
non-decreasing `time` column, built lazily from `|>`-composable pipelines.

**DESIGN.md is the source of truth for the design and must be kept in sync
with any API or semantics change.**

## Commands

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Add a dependency: `julia --project -e 'using Pkg; Pkg.add("Name")'`
  (never hand-edit UUIDs; test-only deps also need an `[extras]` entry)

## Architecture

- `src/context.jl` — `Context{T}`: time window, generic ordered time type
- `src/frame.jl` — `CausalFrame{T}`: opaque, backed by a vector of
  time-disjoint DataFrame chunks; invariants checked in the inner
  constructor; Tables.jl interface
- `src/pipeline.jl` — `CausalPipeline` (lazy `Context -> CausalFrame`),
  `load`
- `src/operators.jl` — sources return a `CausalPipeline`; transforms are
  curried (`filterrows(pred)` returns `CausalPipeline -> CausalPipeline`)
  so both chain with `|>`

## Invariants and conventions

- Sources clip to the half-open interval `[start, stop)`; frames tolerate
  the closed interval `[start, stop]` (intermediate ops may emit at `stop`).
- Every operator must be *causal*: output at time `t` depends only on input
  rows with time `<= t`. This guarantees the chunk-concatenation property
  that streaming will rely on.
- Never expose the backing DataFrames of a `CausalFrame`; `DataFrame(frame)`
  copies. Empty chunks are dropped at construction.
- Naming is Julian: lowercase, no camelCase, no shadowing of Base functions
  (`filterrows` not `filter`, `emptyframe` not `empty`).
