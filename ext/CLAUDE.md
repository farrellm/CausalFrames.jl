# CausalFrames.jl — parquet backend extensions

These two files are the only code in the package that names DuckDB or
Parquet2; `src/parquet.jl` holds the API, the docstrings and the
backend-independent logic and names neither. Which backend is preferred in
which direction, and why, is in `src/CLAUDE.md`'s `parquet.jl` entry — this
file is the contract an extension implements and the traps in implementing it.

## The hook contract

Three `Val{:backend}`-dispatched methods per extension, whose generic
fallbacks in `src/parquet.jl` throw the `READHINT`/`WRITEHINT` message:

- `CausalFrames.backendloaded(::Val{:name}) = true` — how `resolvebackend`
  discovers the extension. It runs eagerly at operator construction *and*
  again per run, so a missing backend is reported where the user typed the
  operator, while one loaded afterwards still counts
- `parquetproducer(::Val{:name}, ctx, path, time, rename)` → a zero-argument
  callable returning the next chunk, or `nothing` at the end
- `parquetsink(::Val{:name}, path, queue, rowgroupsize, opts)` → a `ChunkSink`
  wrapping a `chan -> writeloop(chan, ...)`. Translate options *here*, on the
  pipeline's own task, so an unsupported one is reported when the run starts
  rather than from inside the writer task

## Writing a producer

- It must return only **non-empty** chunks — loop until a clip leaves rows or
  the source is done, since downstream code may assume a chunk has rows
- Per-pull state lives in mutable struct fields, never in reassigned closure
  captures (those get boxed). The dynamically typed fields (`time`, `rename`,
  the reader handle, `prevtime`) are per-chunk *setup* state; per-row work goes
  through `clipchunk!`, which puts it behind a function barrier
- `clipchunk!(df, time, rename, path, "parquet file", prevtime, start, stop)`
  returns `(clipped, sawstop, prevtime)`: carry `prevtime` to the next pull
  (that is what catches a cross-chunk sortedness violation) and stop the stream
  on `sawstop`, a time `>= stop` having been seen
- Every chunk is clipped on arrival whether or not the window was pushed down,
  which is what makes skipping a pure optimization

## Skipping never fails a read

Both backends resolve every uncertainty toward reading *more*, never toward an
error — a wrong answer is unacceptable, a slower one is fine:

- `timesourcename` returns `nothing` when the time column cannot be pinned
  down (a `time` function, an unmatched name, an ambiguous `rename`), which
  simply disables pushdown
- DuckDB catches a time type it cannot bind or compare and re-runs the query
  unfiltered; Parquet2 catches an incomparable statistic and sets
  `usestats = false` for the rest of the run
- Missing or unusable row-group statistics read as `:overlaps`

## DuckDB specifics

- One process-wide in-memory `DuckDB.DB`, created on first use behind
  `DBLOCK` and **never at precompile time**, plus a fresh connection per
  pipeline run: a connection is single-consumer, and a pipeline may be run
  more than once (a self-join reads its file twice)
- The write fallback exists because DuckDB cannot append row groups to a
  parquet file: the stream is staged in a `TEMP TABLE` (which DuckDB spills
  under memory pressure) and written by one `COPY` at the end. An empty stream
  must still leave a valid file — hence the `SELECT NULL::BIGINT AS time WHERE
  FALSE` source
- `compression_codec` is the only `writeparquet` option this backend can
  express. The rest are Parquet2's, and are rejected with an `ArgumentError`
  rather than silently dropped, since dropping would hide a deliberate request

## Parquet2 specifics

- The sink defaults `compute_statistics = ["time"]` unless the caller set it:
  those statistics are exactly what the *other* backend-independent read path
  consults, so the two directions are coupled through the file
- Chunks are only ever merged into a row group, never split, so
  `rowgroupsize = 1` gives one row group per incoming chunk
- The file becomes readable only when `finalize!` writes the footer as the
  channel closes — there is no usable prefix mid-run

## Infrastructure

- **No precompile workload.** `src/precompile.jl` cannot touch the parquet
  paths, because a weak dependency is not loadable at precompile time — one of
  the two documented exceptions to the root CLAUDE.md rule that a new operator
  adds a workload path (the MLJ operators, below, are the other)
- `test/runtests.jl` loads both packages, so `backend = :duckdb`/`:parquet2`
  covers all four read/write combinations in one process. The cases that
  depend on a backend being *absent* — no backend loaded, exactly one loaded —
  have to run in subprocesses, via `insubprocess` in `test/parquet.jl`
- A new backend is wired in `Project.toml` under `[weakdeps]` and
  `[extensions]`, plus `[extras]`/`[targets]` so the tests can load it; the
  module name must match both the file name and the extension key

## The MLJ extension

`CausalFramesMLJModelInterfaceExt.jl` is the only code naming
MLJModelInterface. `src/models.jl` holds the operators (`applymodels`,
`addpredictions`, `modelreports`), their docstrings and the hook fallbacks, and
`src/summarizers.jl` holds `FitModel` and `FittedModel`. The trigger is
MLJModelInterface — not MLJ or MLJBase — because every model package depends on
it, so loading any model loads the extension.

- Five hooks, each with a method for `MMI.Model`:
  - `ismodel` → `true`. `FitModel`'s constructor consults it eagerly, and uses
    `Base.get_extension` to tell "extension not loaded" from "not a model"
  - `fitmodel` → `(fitresult, report)`, from `MMI.fit` over
    `MMI.reformat(model, X, y)...`. The report is passed through
    `MMI.report(model, Dict(:fit => report))`, which is exactly how MLJBase's
    `report(mach)` builds a freshly fit machine's; that is what makes
    `modelreports` agree with `report(mach)` (the raw fit report does not: an
    empty one is `NamedTuple()` there, `nothing` in `report(mach)`)
  - `predictmodel` → the operation (mapped from its symbol by `predictop`)
    over `MMI.reformat(model, X)...`
  - `savefitresult`/`restorefitresult` → `MMI.save`/`MMI.restore`, but only
    where a method is `applicable`: MLJModelInterface declares them with *no*
    methods (the identity fallbacks are MLJBase's), so an unconditional call
    fails for any model that does not implement them. `FittedModel`'s custom
    `serialize`/`deserialize` route through these hooks
- `using MLJModelInterface` (and `using MLJ`) re-exports the scientific types,
  whose `Count` clashes with the `Count` summarizer — tests `import` it, and
  users write `CausalFrames.Count()`
- The model-level API, never machines, so MLJBase is not a dependency even
  weakly. But without MLJBase, MLJModelInterface runs in its *light mode*,
  where `MMI.matrix`, `MMI.table` and the scitype helpers throw. Real model
  implementations call them, so users need `using MLJ`; test models must use
  `Tables.matrix` instead (see `test/models.jl`)
- `predict_mean`, `predict_mode` and `predict_median` are stubs in
  MLJModelInterface, and their `mean.(predict(...))` fallbacks live in MLJBase.
  An unsupported operation therefore surfaces as MLJ's own MethodError
- No precompile workload, for the parquet reason: the weak dep cannot load at
  precompile time, and without it no `FittedModel` exists to feed
  `modelreports` either
