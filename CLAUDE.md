# CausalFrames.jl

Julia package for time-series tables: DataFrames with a monotonically
non-decreasing `time` column, built lazily from `|>`-composable pipelines.

**DESIGN.md is the source of truth for the design and must be kept in sync
with any API or semantics change.**

## Commands

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'` (includes Aqua
  and, on released Julia versions, targeted JET checks in `test/jet.jl`).
  Test files aren't standalone — they rely on `runtests.jl`'s imports and
  `test/fixtures.jl`, so `julia --project test/rolling.jl` won't run
- Build docs: `julia --project=docs docs/make.jl` (one-time setup:
  `julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`).
  Documenter runs strict — an unregistered docstring fails the docs CI job
- Benchmark: `julia --project=benchmark benchmark/benchmarks.jl` (same
  one-time `Pkg.develop(path=".")` setup; defines `SUITE` for PkgBenchmark)
- Formatting is automatic: a `Stop` hook in `.claude/settings.json` formats
  modified `.jl` files once per turn (config in `.JuliaFormatter.toml`; CI
  pins JuliaFormatter v2, so don't format with a v1 install)
- Run one Julia process at a time — concurrent test/docs/benchmark runs race
  on the precompile cache and fail transiently
- Add a dependency: `julia --project -e 'using Pkg; Pkg.add("Name")'`
  (never hand-edit UUIDs; test-only deps also need an `[extras]` entry)
- CI tests Julia 1.10 (minimum supported), 1.12, and pre-release — don't
  use post-1.10 language/stdlib features
- The default branch is `master`, not `main` — target PRs there

## Architecture

Per-module design rationale lives in `src/CLAUDE.md` (loaded when working
under `src/`); DESIGN.md's "Module layout" table is the canonical index.
The parquet backends live in `ext/` behind weak deps, with the backend
contract in `ext/CLAUDE.md`. `notes/` holds investigation records —
measurements and rejected designs, kept so they aren't re-derived, and
explicitly *not* design law.

## Adding an operator or summarizer

A new source or transform touches, in the same commit: `src/CausalFrames.jl`
(include + export), `docs/src/api.md` (register the docstring or the docs job
fails), `DESIGN.md` (module table, export list, semantics), `src/precompile.jl`
(a workload path — the parquet operators are the sole exception, see
`ext/CLAUDE.md`), `test/runtests.jl` (include the new test file), and
`README.md`'s operator table. Nothing enforces that last row — no test, no CI
job — so it is the one that silently drifts; check it before you call the
commit done.

A new summarizer instead touches `src/summarizers.jl` (the type, its state, and
which structured subtype it claims — a performance decision, not a taxonomy
one), `docs/src/api.md`, `DESIGN.md`'s "Summarizers" section and export list,
`test/summarizers.jl`, and `README.md`'s summarizer paragraph. That paragraph is
prose rather than a table, so it drifts even more quietly than the operator
table — check that every exported summarizer still appears there.

## Invariants and conventions

- Sources clip to the half-open interval `[start, stop)`; frames tolerate
  the closed interval `[start, stop]` (intermediate ops may emit at `stop`).
- Every operator must be *causal*: output at time `t` depends only on input
  rows with time `<= t`. This guarantees the chunk-concatenation property
  that streaming will rely on. The sole escape hatch is the `Acausal`
  submodule (`src/acausal.jl`), never re-exported — a forward-looking
  operator goes there, reached only via `using CausalFrames.Acausal`.
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
