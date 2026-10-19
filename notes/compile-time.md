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

## Rejected

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

Most of `rolling.jl`'s remaining 260 s is the kernels `rollsegment!` (about
84 s over 525 instances) and `leadinsegment!` (about 32 s), one instance per
tested summarizer set, window set, schema and key. The tests exercise many
such combinations by design. The other window transforms
(`summarizewindows`, `barwindow`) still specialize their drivers on the
config, the pattern removed from `addrollingcolumns` here.
