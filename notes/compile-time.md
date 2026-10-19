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

## `intervalize`

In its trace before this pass, `load` was 3.5 s of the file's 14 s: about
250 ms per summarizer set, with the closure capturing `protos`, `grouping`
and `outs`, and the whole driver compiling under it. `foldintervals!`, the
kernel, was 3.0 s.

| Change | Harness, 6 fresh `intervalize` calls (two runs each) |
|---|---|
| Before | 6.35–6.55 s |
| The chunk step and flush read an `IntervalConfig` through `IntervalizeState`'s abstract field, taken `@nospecialize`, as in `summarizewindows` | 5.68–5.70 s |
| `preparechunk!` takes `protos` and `grouping` `@nospecialize` (shared with `summarize`, `summarizecycles` and `addsummarycolumns`) | 5.42–5.43 s |

That is 17% off the harness, with identical outputs. `intervalize.jl` on its
own, two runs each: 13.4–13.5 s wall (13.0–13.1 s compile) before, 12.3–12.4 s
(11.9–12.0 s) after, −8%. The second change is
sound for every caller: its results go into `SummaryFold`'s untyped fields,
and its column table is untyped too (it comes from a `DataFrame`), so each
caller already reached its kernel by a dispatch per chunk. What remains in the
harness is `foldintervals!` (3.0 s over 7 instances) and the generated
`rowsframe` unzip, one per output row type (about 0.1 s each for 15 columns).

Runtime: over the `intervalize` and `summarize` groups, the geometric mean of
3 alternating rounds against the parent commit was 0.997. Allocations rose
by 3–9 per run (dispatches per chunk). A second run put `intervalize/keyless`
at +4.7%, with the `rowwise/filterrows` control at +2%. Re-run with the
current side first in each round, the same three came out at 0.983, 0.991 and
0.996. The deficit followed the run position, not the code, so alternate the
order as well as the sides before believing a few percent.

## `summarizecycles`

In `test/summarize.jl`'s trace, `summarizecycles`' `load` entries were 1.13 s
over 15 calls (about 75 ms each), and its kernel `foldcycles!` was 2.9 s.
`summarize` (2.9 s over 22) and `addsummarycolumns` (0.7 s over 10) have the
same closure pattern; they follow.

| Change | Harness, 6 fresh `summarizecycles` calls (two runs each) |
|---|---|
| Before | 5.46–5.71 s |
| The chunk step and flush (`cyclechunk!`, `cycleflush`) read a `FoldConfig` through `FoldRun`'s abstract field and take it `@nospecialize` | 4.97–5.00 s |

The outputs are identical. `test/summarize.jl` on its own barely moves
(21.1–21.3 s wall before, 20.9–21.1 s after, two runs each): its 15
`summarizecycles` calls were about 1.1 s of `load` to begin with. What
remains in the harness is `foldcycles!`
(3.1 s over 14 instances), the `rowsframe` unzip (0.65 s), and `newstates`
plus `newgroups` (0.4 s, once per run, shared by every summarizing transform;
they compute the state types, so there is nothing to erase).

Runtime: `summarize/cycles`, `cycles-keyed`, `keyless` and `keyed` plus the
`rowwise/filterrows` control, 4 alternating rounds with the side that goes
first swapped each round: geometric mean 0.998, every range overlapping,
allocation counts unchanged.

## `summarize`

The same erasure, reusing `summarizecycles`' `FoldConfig` and `FoldRun`
(`keyed` is `!isempty(fold.keycols)`, so the config needs nothing more).

| Change | Harness, 6 fresh `summarize` calls |
|---|---|
| Before (two runs) | 5.08–5.50 s |
| `summarizechunk!` and `summarizeflush` read the config through `FoldRun`, `@nospecialize`; the keyed emission moves behind a typed barrier, `sortedrows` (three runs) | 4.30–4.39 s |
| The keyless flush's one row goes through `rowsdataframe` instead of `DataFrame([row])`, which compiled Tables.jl's path per row type (three runs) | 3.96–4.11 s |

The outputs are identical. `test/summarize.jl` on its own: 20.9–21.3 s wall
before, 19.4–19.9 s after (−7%), two runs each. Before the change, the keyed flush's comprehension ran
over `fold.groups`, which is untyped, so it was inferred abstractly. Behind
`sortedrows` it is concretely typed. `lastrow`'s flush still has a one-row
`DataFrame([row])`.

Runtime: the whole `summarize` group plus the control, 4 alternating rounds
with the first side swapped: geometric mean 0.984, every range overlapping;
allocations 1–6 fewer per run.

## `addsummarycolumns`

The same erasure through `FoldRun`, with `runningchunk!` the driver. The
output names `preparechunk!` checks as new columns are read back from
`cfg.outs`, so `FoldConfig` needs no extra field.

| | Harness, 6 fresh `addsummarycolumns` calls (three runs each) |
|---|---|
| Before | 3.58–3.73 s |
| After | 3.36–3.45 s |

The outputs are identical. `test/summarize.jl` moves within noise, 19.0–19.1 s
before and 18.8–18.9 s after, alternating. Two runs of each side back to back
had first put it 0.8 s *slower*, the current side's block run straight after
a full `Pkg.test()`: time a file alternating too, not in blocks. It gains less than the others: the driver was
small, and what remains is `foldrunning!` (1.9 s over 7 instances) and the
`rowsdataframe` unzip (0.5 s). Runtime: `summarize/running` (the
`addsummarycolumns` benchmark), 6 alternating rounds with the first side
swapped: 1.001, with the control at 0.996, and 9 more allocations per run.

## `lastrow`

`lastrow` has no summarizers, so its closure captured only `Val(keynames)`,
and the driver compiled under `load` once per key-name tuple (100–200 ms
each in `test/lastrow.jl`'s trace). Its flushes built frames with
`DataFrame(rows)`, about 120 ms per output row type.

| Change | Harness, 6 fresh `lastrow` calls (three runs each) |
|---|---|
| Before | 2.43–2.60 s |
| Both flushes build with `rowsdataframe` | 2.35–2.41 s |
| The key names move into `LastRowState` as an abstract `Val` field; `lastrowchunk!` and `lastrowflush!` read them there, and `storekeytype`/`lastsegment!` take them by dispatch | 1.85–1.89 s |

The outputs are identical. `test/lastrow.jl` on its own, alternating: 5.0–5.1 s
wall before, 4.3–4.4 s after (−14%). Runtime: the `lastrow` group and the control, 6
alternating rounds with the first side swapped: geometric mean 0.986, every
range overlapping. Allocations went up 18 per run (1M rows). The control's
count also differs between the two environments (651 against 671), so not
all of that is the change.

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
`intervalize`'s is `foldintervals!`, and `summarizecycles`' is `foldcycles!`.
