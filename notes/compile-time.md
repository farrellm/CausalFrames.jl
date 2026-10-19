# Where the test suite's time goes, and what cut `rolling.jl`'s

This is an investigation record, not design law: `DESIGN.md` is still the source
of truth. It exists so the next person who looks at compile time does not have
to repeat the measurements.

Summary: the suite is almost all compilation. `rolling.jl` was the largest
file, and about a third of its compile time went on setup code (run once per
run or per chunk) that specialized on every summarizer set. Erasing or
narrowing that code cut the file from 352 s to 260 s, with no runtime cost.
The per-row kernels' remaining cost is the price of full specialization:
un-inlining in them saves little compile time and costs runtime.

## Environment

| | |
|---|---|
| Julia | 1.13.1, `JULIA_NUM_THREADS=4` |
| Machine | 8 cores, 15 GB, swap full during the runs (absolute times inflated) |
| Date | 2026-10-18 |

## The suite

`Pkg.test()` took 18.8 min of wall time. About 3 minutes went on the test
environment's precompile, which uses its own cache because of its flags
(`--depwarn=yes` and others). The testset took 15m46s. Timing each `include`
with `@timed` gave 883 s of compile time and 57 s of running. Four files took
68% of the total, each about 98% compile: `rolling.jl` (352 s), `windows.jl`
(127 s), `summarizers.jl` (94 s) and `rowterms.jl` (72 s).

## Tools, and what they show on 1.13

- `SnoopCompileCore.@snoop_inference` (SnoopCompile 3.2) works. On 1.13 its
  frame times appear to include more than inference: the frames summed to
  368 s, more than the 352 s file. Use it for counts and ranking, not for
  absolute numbers. `DataFrames` exports `flatten` too, so call
  `SnoopCompile.flatten`.
- `@snoop_llvm` writes empty files on 1.13.
- `--trace-compile=FILE --trace-compile-timing` is the most useful view. It
  gives one line per dynamically dispatched compile, timed to include LLVM and
  every callee compiled statically under it. That shows which *entry point*
  the cost hangs off.
- `Profile` with `C = true` while the compiler runs split one case (a 19-state
  keyed set) into about 65% of samples in LLVM (codegen and its passes) and
  40% in Julia inference and its optimizer (the two overlap). LLVM is at least
  as large as inference.

A harness of six representative `addrollingcolumns` calls, each a fresh
specialization, gave a 20-second loop instead of a 6-minute one. It wrote its
outputs to disk, so every change was checked for identical frames and column
types.

## What was changed (`rolling.jl` compile time, cumulative)

| Change | Harness total | `rolling.jl` |
|---|---|---|
| baseline | 21.5 s | 352 s |
| `tiering`: generated `tierpick` past a `Val` permutation, replacing `Tuple(src[i] for i … if …)` (a widening `grow_to!` per element type) | 17.8 s | |
| `rolltiers`: barrier past `storekeytype`/`storerowtype`, `ntuple(…, Val(n))` instead of the runtime-`n` 10-way `ntuple` | 16.8 s | |
| `rowsdataframe` replacing `DataFrame(rows)` (Tables.jl's row-to-column path per row type) | 15.8 s | |
| `prototypes`' `expand` `@nospecialize`, with a loop over `dependencies` instead of `foreach` | 13.1 s | 285 s |
| the chunk step through `LeadInGate`'s erased `cfg`, and the per-chunk driver `@nospecialize(cfg)` | 12.6 s | 260 s |

`rowsdataframe` is also faster. Over 200,000 four-field rows it took 1.57 ms
against `DataFrame(rows)`'s 2.30 ms with two `missing`-admitting columns, and
0.95 ms against 1.05 ms without.

## `summarizewindows` and `barwindow`

The same pass over the other two window transforms, starting from the
`addrollingcolumns` changes above.

| Change | Measured on | Before | After |
|---|---|---|---|
| `summarizewindows`' chunk step and flush read `cfg` through `WindowState`'s abstract field, and the per-chunk driver takes it `@nospecialize` | harness, 7 fresh `summarizewindows` calls | 13.4 s | 12.2 s |
| `newbarwindow` (once per state and per widening) takes its arguments `@nospecialize` | `barwindow.jl`, two runs each | 33.1 s | 32.0 s |

Runtime for the first: `windows` group geometric mean 0.984 over 3
alternating rounds against the parent commit, nothing slower; 2–17 more
allocations per run (dynamic dispatches per chunk, not per row). The second
touches only state construction (a key's new state is a `fresh` copy).

Rejected:
- **A barrier past the type values in `preparewindows!`**, as `rolltiers`
  got. The harness didn't move (13.4 s, then 13.7 s). Unlike `rolltiers` it had
  no runtime-length `ntuple`, so there was little abstract code to remove.
- **`@nospecialize(outs::Val{O}) where {O}`** in `newbarwindow`. The static
  parameter brought most of the specialization back (32.7 s against 32.0 s);
  reading `O` through a one-line `valvalue(::Val{X}) where {X} = X` keeps the
  saving.

`barwindow.jl`'s trace is flat. Its biggest entries are the states' own
per-row `update!`/`value` and the test file's helpers, so a few percent is
what this file has to give.

## Per-file timings, before and after

Each test file's `include` timed with `@timed` over a whole run of
`test/runtests.jl`: the first run on `master` before any of this (the
measurement that started it), the second on the branch with every change
above (the skill script `timed_runtests.jl`, `--depwarn=yes` as `Pkg.test`
runs). Both on Julia 1.13.1 with 4 threads on the same machine, but on
different days and under different memory pressure, so a few seconds either
way per file is noise. Files that run all their code early pay for compiling
code shared with later ones.

| File | Before wall | Before compile | After wall | After compile | Change (wall) |
|---|---|---|---|---|---|
| `rolling.jl` | 351.6 | 343.6 | 236.9 | 229.5 | −33% |
| `windows.jl` | 126.9 | 125.1 | 90.4 | 88.8 | −29% |
| `summarizers.jl` | 93.8 | 91.7 | 74.3 | 72.4 | −21% |
| `rowterms.jl` | 71.8 | 70.8 | 47.6 | 46.8 | −34% |
| `parquet.jl` | 47.8 | 29.7 | 43.1 | 26.7 | −10% |
| `barwindow.jl` | 39.2 | 37.0 | 28.3 | 26.2 | −28% |
| `operators.jl` | 30.2 | 29.9 | 26.9 | 26.7 | −11% |
| `jet.jl` | 29.1 | 20.7 | 23.0 | 16.2 | −21% |
| `models.jl` | 23.0 | 21.4 | 18.0 | 16.6 | −22% |
| `keys.jl` | 22.9 | 22.7 | 19.3 | 19.1 | −16% |
| `summarize.jl` | 17.5 | 17.2 | 14.7 | 14.4 | −16% |
| `intervalize.jl` | 12.9 | 12.4 | 10.8 | 10.4 | −16% |
| `merge.jl` | 9.1 | 8.7 | 8.1 | 7.8 | −11% |
| `table.jl` | 8.1 | 7.8 | 7.3 | 7.0 | −10% |
| `sortcycles.jl` | 7.6 | 7.5 | 6.9 | 6.8 | −9% |
| `join.jl` | 6.4 | 6.1 | 5.6 | 5.3 | −13% |
| `acausal.jl` | 5.6 | 5.3 | 5.2 | 4.9 | −7% |
| `fill.jl` | 5.3 | 5.1 | 5.0 | 4.7 | −6% |
| `lookupjoin.jl` | 4.8 | 4.6 | 4.4 | 4.2 | −8% |
| `lastrow.jl` | 3.6 | 3.5 | 3.3 | 3.1 | −8% |
| `frame.jl` | 3.5 | 3.5 | 3.2 | 3.2 | −9% |
| `segtree.jl` | 2.4 | 2.3 | 2.1 | 2.1 | −13% |
| `jls.jl` | 2.0 | 1.8 | 1.8 | 1.7 | −10% |
| `stream.jl` | 1.9 | 1.8 | 1.7 | 1.5 | −11% |
| `context.jl` | 0.3 | 0.3 | 0.3 | 0.2 | |
| **Included files** | **927.9** | **881.1** | **688.7** | **646.7** | **−26%** |

Seconds. Aqua (8.3 s before) and the imports aren't `include`s, so they're
left out of both totals. The files untouched by these changes still drop 6–16%,
more than this table's noise. Part of that is real (`rowsdataframe` and
`prototypes` serve every summarizing transform, and `operators.jl` and
`keys.jl` call them), and part is probably the conditions: the first run had
swap full. The after run hit one error, in `jet.jl`'s `summarizewindows`
kernels testset, which still built `WindowState` without its config. That is
fixed in the commit that changed the constructor, and a full `Pkg.test()` has
since passed, so the `jet.jl` row is short that one testset's body.

## Rejected (`rolling.jl`)

- **A `@noinline` barrier around `summaryvalues(mergestates(…))` in
  `emitroll!`**. The barrier lets windows of one value type share a single
  compiled emission. It saved about 5% of the harness's compile time. In an
  alternating A/B against `master` (3 rounds, minimum taken) it cost
  `rolling/refold` 12% and `rolling/running` 5% at runtime. Reverted, both were
  back at parity. The kernels' inlining is load-bearing.

## Runtime check

The changes were checked with alternating A/B runs of the benchmark suite
against a `master` worktree, minimum over rounds, `-t1`. Over the `rolling`
and `windows` groups the geometric mean was 0.976, and no benchmark was
reproducibly slower. On this machine a single run is noisy by ±5–10%: an
untouched benchmark (`rowwise/sortcycles-column`) moved 8% between two runs,
and an apparent 4.8% slowdown in `windows/tracking-keyed` did not survive a
bisect. Alternate base and current runs and take minimums before believing a
difference.

## What remains

Most of what remains in the window transforms is their kernels: in
`rolling.jl`, `rollsegment!` (about 84 s over 525 instances) and
`leadinsegment!` (about 32 s); in `windows.jl`, `windowrows!` (about 41 s over
152 instances). There is one instance per tested summarizer set, window set,
schema and key, and the tests exercise many such combinations by design.
`intervalize` follows `summarizewindows`' driver pattern and hasn't been
narrowed.
