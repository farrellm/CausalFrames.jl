# CausalFrames.jl

Julia package for time-series tables: DataFrames with a monotonically
non-decreasing `time` column, built lazily from `|>`-composable pipelines.

**DESIGN.md is the source of truth for the design and must be kept in sync
with any API or semantics change.**

## Commands

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'` (includes Aqua
  and, on released Julia versions, targeted JET checks in `test/jet.jl`)
- Format: `julia -e 'using JuliaFormatter; format(".")'` (config in
  `.JuliaFormatter.toml`)
- Add a dependency: `julia --project -e 'using Pkg; Pkg.add("Name")'`
  (never hand-edit UUIDs; test-only deps also need an `[extras]` entry)
- CI tests Julia 1.10 (minimum supported), 1.12, and pre-release — don't
  use post-1.10 language/stdlib features
- The default branch is `master`, not `main` — target PRs there

## Architecture

Per-module design rationale lives in `src/CLAUDE.md` (loaded when working
under `src/`); DESIGN.md's "Module layout" table is the canonical index.

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
