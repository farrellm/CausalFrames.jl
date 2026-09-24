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
  Documenter runs strict — an unregistered docstring fails the docs CI job.
  The home page is **generated from `README.md`** by `docs/make.jl`;
  `docs/src/index.md` is gitignored, so edit the README, never that file
- Doctests (every example is one — see "Documentation and error-message style")
  run only in the docs build. To regenerate outputs, build once with
  `makedocs(; doctest = :fix, …)`, starting from a non-empty placeholder:
  Documenter misplaces output fixed from an empty `# output` section
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
The parquet backends and the MLJ model hooks live in `ext/` behind weak deps,
with their contracts in `ext/CLAUDE.md`. `notes/` holds investigation records —
measurements and rejected designs, kept so they aren't re-derived, and
explicitly *not* design law.

## Adding an operator or summarizer

A new source or transform touches, in the same commit: `src/CausalFrames.jl`
(include + export), its module entry in `src/CLAUDE.md` (the design rationale
and performance constraints), its category page under `docs/src/api/` (its own
`` ## `name` `` header plus a `@docs` block — an unregistered docstring fails
the docs job), a link in `docs/src/api/index.md`, `DESIGN.md` (module table,
export list, semantics), `src/precompile.jl` (a workload path — the parquet and
MLJ operators are the only exceptions, see `ext/CLAUDE.md`), `test/runtests.jl`
(include the new test file), and the matching `README.md` operator table, the
name linking to that header.

A new summarizer instead touches `src/summarizers.jl` (the type, its state, and
which structured subtype it claims — a performance decision, not a taxonomy
one), `docs/src/api/summarizers.md` (again its own header),
`docs/src/api/index.md`, `DESIGN.md`'s "Summarizers" section and export list,
`test/summarizers.jl`, and `README.md`'s summarizer table.

A new keyword on the file sources (as `closed` and `skipmissing` were) threads
through the shared clip: `clipchunk!`/`gatherchunk!` in `src/operators.jl`, the
`parquetproducer` hook's positional signature (the fallback in `src/parquet.jl`
and both `ext/` producers, with `ext/CLAUDE.md`'s hook contract), plus
`readjls`/`readtable` wherever the option applies. Then update DESIGN.md and
the README's source descriptions, and test every parquet backend.

The `## `name`` header, the overview link and the README link are checked by
`checkapilinks` in `docs/make.jl`: a missing one, or a README link naming the
wrong API page, fails the docs build. That is the only thing holding the README
tables in sync, so keep the convention — one code-span `##` header per operator
or summarizer, prose headers for everything else.

## Documentation and error-message style

Docstrings of public operators and summarizers follow one reference layout
(`forwardfill` in `src/fill.jl`, `readtable` in `src/table.jl` and `Variance` in
`src/summarizers.jl` are models):

- The signature block lists every positional and keyword argument with its
  default, and the return type. A transform shows both forms,
  `op(args...; kw = d) -> (CausalPipeline -> CausalPipeline)` and
  `op(p::CausalPipeline, args...; kw = d) -> CausalPipeline`; a source returns
  `CausalPipeline`, a summarizer `Name(column::Symbol) -> Summarizer`.
- A short summary follows, saying what it is ("A transform …", "A source …").
  A summarizer names its output column (`` in `:{column}_mean` ``) and its value
  for no rows.
- `# Arguments` and `# Keywords` bullets, as `` - `name = default`: … ``, cover
  *every* argument: its constraints and which bad values are an
  `ArgumentError`. Leave a heading out only when there's nothing to put under it.
- Then a `jldoctest` example. Notes on rarer errors (chunk-shape changes, say)
  go after it. Link other names with ``[`name`](@ref)``.
- Leave out implementation rationale, which lives in DESIGN.md and
  `src/CLAUDE.md`. Don't write "curried form" boilerplate either:
  `docs/src/api/index.md` explains currying and the pipeline-first form once.
- Don't put a comment between a docstring and its definition, because the
  comment detaches the docstring.

Examples in docstrings, `docs/src/recipes.md` and the README are all doctests
with a `# output` section:

- Each docstring or Recipes example is self-contained. It builds data with
  `readtable` over an inline `DataFrame`, writes files under `mktempdir()`, and
  shows its result, usually with `DataFrame(load(Context(…), p))`.
- README examples are plain ```` ```julia ```` blocks, because GitHub renders
  those. `docs/make.jl` turns them into one shared `jldoctest readme` session,
  so a later README block may use an earlier block's bindings.
- An API page opens with a short intro paragraph. It then has one
  `` ## `name` `` header and `@docs` block per public name.

Error messages:

- Start with the operator name, either as `op <argument> …`
  (`forwardfill tolerance must be non-negative, got -1`) or as `op: …` for
  data errors. A shared validator takes the calling operator's name, as
  `prototypes(…, "summarize")` in `src/summarize.jl` does, so the message names
  the transform the user called.
- Show column names with `repr` (`:a`, `:time`). Use the phrasings
  `must be X, got Y`, `requires at least one X`, `must be unique`,
  `op key column :k not found in the input` and
  `invalid op spec of type T: expected …`.
- Put code in hints in backticks (`` pass `skipmissing = true` ``), and don't
  end a message with a period.

## Invariants and conventions

- Sources clip to the half-open interval `[start, stop)` unless a `read*`
  source is passed `closed = true`; frames tolerate the closed interval
  `[start, stop]` (intermediate ops may emit at `stop`).
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
