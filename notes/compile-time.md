# Where the test suite's time goes, and what cut `rolling.jl`'s

This is an investigation record, not design law: `DESIGN.md` is still the source
of truth. It exists so the next person who looks at compile time does not have
to repeat the measurements.

Summary: the suite is almost all compilation. `rolling.jl` was the largest
file, and about a third of its compile time went on setup code (run once per
run or per chunk) that specialized on every summarizer set. Erasing or
narrowing that code cut the file from 352 s to 260 s, with no runtime cost.
The same erasure over every other transform's driver took the included test
files from 928 s to 671 s in all.
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
output names `preparechunk!` checks as new columns now sit in `SummaryFold`'s
`newcols` field (empty for the other transforms), not in `FoldConfig`.

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

## The config moves into `SummaryFold`

The four sections above describe each step as measured, and so name types
that are gone. A last change (a6e6595) moved `FoldConfig` into `SummaryFold`
itself, as an abstract `cfg` field: `FoldRun` went, and so did
`IntervalConfig` (`FoldConfig` plus `closelast`, now a plain field of
`IntervalizeState`). `preparechunk!` now reads `protos`, `grouping` and the
new-column names from the fold rather than taking them `@nospecialize`. The
four harnesses were unchanged within noise, with identical outputs;
`addsummarycolumns` stopped unwrapping its output names every chunk (8 fewer
allocations per run in `summarize/running`), and the `summarize` and
`intervalize` groups came out at a geometric mean of 1.0006.

## The as-of join engine

`asofjoin`, `applymodels` and `Acausal.futurejoin` share one driver, and its
closure captured `JoinConfig{KN,Tol,C,D}`, so `joinchunk!`, `pullright!`,
`matchchunk!`, `checknames` and `assemble` compiled once per key, tolerance
type, strictness and direction. The config moves into `JoinState` as an
abstract field, read back by the closure, and the driver takes it
`@nospecialize`, as in `addrollingcolumns`. `js.store` was already untyped,
so `segment!` was already reached by a dispatch per call; nothing is added on
the hot path.

| | Harness, 6 fresh `asofjoin`/`futurejoin` calls (two runs each) |
|---|---|
| Before | 4.37–4.39 s |
| After | 2.63–2.77 s |

The outputs are identical. `test/join.jl` on its own, alternating: 7.7–7.8 s
before, 7.1–7.3 s after; `test/acausal.jl` 5.7–5.8 s before, 4.8 s after.
Runtime: the `join` group and the control, 6 alternating rounds with the first
side swapped: the four as-of and future joins at 1.004–1.015, ranges
overlapping, 8 fewer allocations per run. `lookup-drop` read 1.04 (1.06 in a
first 4-round run) although `lookupjoin` was untouched, which is this
machine's noise at that size.

## `forwardfill`

`lastrow`'s shape before its change: the closure captured `Val(keynames)` and
`tolerance`, so `fillchunk!` and the iterator compiled once per key and
tolerance type. Both move into `ForwardFillState` as abstract fields.
`fillprepared!` was already reached by a dispatch per chunk (on
`st.fillval`), so the kernels see the same concrete types.

| | Harness, 6 fresh `forwardfill` calls (two runs each) |
|---|---|
| Before | 2.27–2.30 s |
| After | 1.54–1.55 s |

The outputs are identical. `test/fill.jl` on its own, alternating: 7.4–7.5 s
before, 7.0–7.1 s after. There is no `forwardfill` benchmark; over 1M rows,
three alternating runs each, minimums 9.12 ms against 9.24 ms keyless and
28.68 ms against 28.07 ms keyed (two filled columns, `tolerance = 10`), with 6
fewer allocations.

## Per-file timings, before and after

Each test file's `include` timed with `@timed` over a whole run of
`test/runtests.jl`: the first run on `master` before any of this (the
measurement that started it), the second at 00c29d8 (the window transforms'
pass), the third at the branch head with every change above (the skill script
`timed_runtests.jl`, `--depwarn=yes` as `Pkg.test` runs). All on Julia 1.13.1
with 4 threads on the same machine, but on different days and under different
memory pressure, so a few seconds either way per file is noise. Files that run all their code early pay for compiling
code shared with later ones.

| File | Before wall | Before compile | 00c29d8 wall | Head wall | Head compile | Change (wall) |
|---|---|---|---|---|---|---|
| `rolling.jl` | 351.6 | 343.6 | 236.9 | 239.8 | 232.4 | −32% |
| `windows.jl` | 126.9 | 125.1 | 90.4 | 88.1 | 86.6 | −31% |
| `summarizers.jl` | 93.8 | 91.7 | 74.3 | 67.2 | 65.3 | −28% |
| `rowterms.jl` | 71.8 | 70.8 | 47.6 | 44.2 | 43.3 | −38% |
| `parquet.jl` | 47.8 | 29.7 | 43.1 | 43.2 | 26.8 | −10% |
| `barwindow.jl` | 39.2 | 37.0 | 28.3 | 28.0 | 26.0 | −29% |
| `operators.jl` | 30.2 | 29.9 | 26.9 | 26.8 | 26.7 | −11% |
| `jet.jl` | 29.1 | 20.7 | 23.0 | 23.8 | 16.5 | −18% |
| `models.jl` | 23.0 | 21.4 | 18.0 | 17.9 | 16.5 | −22% |
| `keys.jl` | 22.9 | 22.7 | 19.3 | 17.6 | 17.4 | −23% |
| `summarize.jl` | 17.5 | 17.2 | 14.7 | 12.1 | 11.8 | −31% |
| `intervalize.jl` | 12.9 | 12.4 | 10.8 | 9.1 | 8.7 | −29% |
| `merge.jl` | 9.1 | 8.7 | 8.1 | 8.1 | 7.7 | −11% |
| `table.jl` | 8.1 | 7.8 | 7.3 | 7.4 | 7.1 | −9% |
| `sortcycles.jl` | 7.6 | 7.5 | 6.9 | 7.0 | 6.9 | −8% |
| `join.jl` | 6.4 | 6.1 | 5.6 | 5.2 | 4.8 | −19% |
| `acausal.jl` | 5.6 | 5.3 | 5.2 | 4.2 | 3.9 | −25% |
| `fill.jl` | 5.3 | 5.1 | 5.0 | 4.6 | 4.3 | −13% |
| `lookupjoin.jl` | 4.8 | 4.6 | 4.4 | 4.5 | 4.3 | −6% |
| `lastrow.jl` | 3.6 | 3.5 | 3.3 | 2.6 | 2.5 | −28% |
| `frame.jl` | 3.5 | 3.5 | 3.2 | 3.1 | 3.0 | −11% |
| `segtree.jl` | 2.4 | 2.3 | 2.1 | 2.2 | 2.1 | −8% |
| `jls.jl` | 2.0 | 1.8 | 1.8 | 1.8 | 1.7 | −10% |
| `stream.jl` | 1.9 | 1.8 | 1.7 | 1.6 | 1.5 | −16% |
| `context.jl` | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 |  |
| **Included files** | **927.9** | **881.1** | **688.7** | **670.6** | **628.2** | **−28%** |

Seconds. Aqua (8.3 s before) and the imports aren't `include`s, so they're
left out of the totals. The files untouched by these changes still drop 6–16%,
more than this table's noise. Part of that is real (`rowsdataframe` and
`prototypes` serve every summarizing transform, and `operators.jl` and
`keys.jl` call them), and part is probably the conditions: the first run had
swap full. The 00c29d8 run hit one error, in `jet.jl`'s `summarizewindows`
kernels testset, which still built `WindowState` without its config. That is
fixed in the commit that changed the constructor, and a full `Pkg.test()` has
since passed, so its `jet.jl` figure is short that one testset's body. The
head run passed every test.

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

The branch head against `master`, the whole suite (89 benchmarks), 4
alternating rounds: geometric mean 0.993. `rolling` is mostly 2–8% faster.
The largest apparent slowdown, `join/lookup-keyed` at 1.12, is in code the
branch doesn't touch. The harness itself is biased: `ab_bench.sh` runs the
base from a worktree with its own benchmark environment and the current side
from the repository, and HEAD against HEAD put `join/asof` at 1.048 with no
overlap. Run instead through two worktrees with identically built
environments (6 rounds, alternating), `master` against the head gave
`lookup-keyed` 1.083 from one fast base round (base 23.2–27.6 ms, head
25.0–26.5 ms, ranges overlapping), `lookup-drop` 1.027, `asof` 0.976,
`asof-keyed` 1.009 and `summarize/countdistinct` 1.000. For a few percent,
compare two worktrees, not a worktree against the repository.

## What remains

Most of what remains in the window transforms is their kernels: in
`rolling.jl`, `rollsegment!` (about 84 s over 525 instances) and
`leadinsegment!` (about 32 s); in `windows.jl`, `windowrows!` (about 41 s over
152 instances). There is one instance per tested summarizer set, window set,
schema and key, and the tests exercise many such combinations by design.
`intervalize`'s is `foldintervals!`, and `summarizecycles`' is `foldcycles!`.

## The idiom refactor (2026-10-21)

A clarity refactor, four commits: the summarizing kernels written once over
their groups, the window tiers' shared eviction, `outname` and an abstract
`DerivedState`, and shared producer/argument helpers. It was gated on compile
time as well as runtime. Julia 1.13.2, 4 threads for the suite, `-t1` for the
harness and benchmarks.

| | `master` | branch |
|---|---|---|
| Included test files, compile | 643.1 s | 628.0 s |
| Included test files, wall | 687.4 s | 671.0 s |
| Harness (14 first calls), compile | 10.17–10.28 s | 10.13 s |

The harness is `load` of 14 representative transforms in a fresh process, each
its own specialization (summarize, summarizecycles keyless/keyed/dense,
addsummarycolumns, intervalize keyless/keyed/dense, summarizewindows, two
addrollingcolumns, lastrow, asofjoin), with the outputs serialized and
compared with `master`'s, which they matched exactly. The suite's 15 s drop
is mostly `rolling.jl` (243 → 231 s); the rest moved by less than noise.
The benchmark's `chain8-keyed`, the case that once caught a type-nesting
blow-up, compiles its first `load` in 2.27–2.31 s against `master`'s
2.27–2.35 s (three fresh processes each, alternating).

At runtime, a full-`SUITE` A/B of two worktrees (4 rounds, alternating order,
minimums) gave a geomean ratio of 1.004 over 89 benchmarks. No allocation
count rose: every `rolling`, `windows` and `join` benchmark allocates 9–285
fewer per run. Every flag past 2% (`rowwise/filterrows` ×1.09, and
`lastrow/keyless`, `sources/readtable-frame` and `rowwise/pipeline` near ×1.03)
re-measured in isolation over 6 rounds at ×1.000–1.018, which is within noise.

### Rejected: `compsub` as `compadd` of the negated term

`compsub(a, x) = isfinite(x) ? compadd(a, -x) : compcount(a, x, -1)`, and
`compsubpair` as `compaddpair(s, -p, -e)`, are bit-identical to the written-out
forms (negating a float is exact, and `c + -e` is `c - e`). But a keyed
`summarize(Variance(:x))` went from 1.0 to 3.4 s of compile, and the harness
from 10.1 to 13.6 s (an addrollingcolumns over `Variance` and `Quantile` from
1.3 to 2.4 s). Reverting `compensated.jl` alone restored both. The cause
wasn't traced; the co-moment state calls these several times per row, so a
larger inlined body at each subtraction site is the likely suspect. Kept
written out.

## The monoid tier's two-stack queues (2026-10-22)

The monoid tier moved from per-key segment trees to Two-Stacks Lite queues
(`src/monoidqueue.jl`). This is a runtime change, gated on compile time. Julia
1.13.2, 4 threads for the test files, `-t1` for the harness.

| | `master` | branch |
|---|---|---|
| `rolling.jl` + `windows.jl` + `barwindow.jl` + `jet.jl` + the structure's own test file, compile (two runs each, alternating) | 381.7, 374.0 s | 378.6 s (final) |
| `rolling.jl` alone | 242.2, 235.9 s | 239.1 s |
| Harness (8 fresh monoid-tier calls: `Product` keyless, keyed with 3 windows, mixed, lead-in, `Bars`; `summarizewindows` keyless, keyed, dense), compile | 9.9–10.2 s | 10.2–10.5 s |

The harness spread overlaps, and its `--trace-compile-timing` totals matched
exactly (11.7 s over 465 entries on both sides), so this is noise. Outputs
agreed with `master`'s apart from `Product` over floats, which differed by at
most 1.4e-15 relative: the queue associates the product differently from the
tree.

Two things mattered for compile time:

- With the window count `W` as a type parameter (`NTuple{W,PartialQueue}`),
  the harness's lead-in case compiled 0.2 s slower than `master`. It shares
  its key and row type with a three-window case, and `SegTree{S,R,T}` let it
  reuse that case's methods, while `MonoidQueues{S,R,T,W}` didn't. Holding the
  queues in a `Vector` removed the gap.
- `@inline` on `queuepush!` or `livestates` made no measurable difference
  either way.

At runtime (two worktrees, 4 alternating rounds, `rolling` and `windows`
groups, minimums), the `rolling` monoid-tier entries halved: `tree` 8.0 →
4.0 ms, `bars-tree` 7.7 → 3.8 ms, `leadin-wide-tree-keyed` 33.3 → 16.8 ms
(`-tworuns` 34.5 → 20.0 ms). `windows/tree` came out at ×1.027, with
overlapping ranges, while allocating 21,141 times against 175,905. The other
39 benchmarks were flat (geomean 0.942 over all 41). The peak live heap of
`leadin-wide-tree-keyed` (`benchmark/peakmemory.jl`) fell from 12.2 to 8.7 MiB.

## Shared runs (issue #105), 2026-10-25

Pipelines became DAG nodes (`src/share.jl`): a run function no longer
captures its inputs, which are an erased `Vector{CausalPipeline}`, and the
planner opens each input through `openinput`, inferred `Any`. The first cut
specialized everything that takes a pipeline (the builders, the planner's
`instances`/`postorder!`, `load`) on each node's `CausalPipeline{F}`: about
80 ms of compile per node type, `|>` included. `operators.jl` + `join.jl`
went from 37 s to 56 s, and the included files from 797 s to about 890 s (the
new `sharing.jl` aside). A `--trace-compile-timing` of one five-stage chain
showed it: five `instances` entries at 80 ms each, against a single 220 ms
`load` on master.

Taking pipelines `@nospecialize` throughout that setup code (and replacing the
builders' closures with small callable structs: `SourceRun`, `UnaryRun`,
`UnaryTransform`, `PassRun`) fixed it. The chunk iteration still compiles per
iterator type, as before, but each operator's iterator is now built behind
a dynamic `openinput`, so no evaluation infers the whole chain at once.

| | master | branch |
|---|---|---|
| 8-call harness (fresh pipelines), compile | 4.35 s | 3.87 s |
| `operators.jl` + `join.jl`, wall | 37.1 s | 35.0 s |
| included test files, wall (compile) | 797.0 s (747.3) | 747.8 s (700.6), with the new `sharing.jl` (20.5 s) |
| `rolling.jl` | 283.7 s | 253.0 s |

Julia 1.13.2, 4 threads, the per-file table from `timed_runtests.jl`; the
harness outputs were identical, element types included.

The `/simplify` pass over it (abandoning moved into the plan's `InstanceRun`,
`joinnode`, `SideClip` over `LeadInDrop`) was flat: included files 762.1 s
wall (707.7 compile) before, 767.8 s (712.2) after, every file within noise.
The one outlier, `barwindow.jl` (+4.7 s), measured 32.0–32.9 s compile on both
sides when timed alone, alternating, twice each.

## Issue #141 evaluated (2026-11-02)

Issue #141 proposed widened Float16/Float32 kernels, more chunk-setup
barriers in `rolling.jl`/`windows.jl`, and a larger precompile workload. It
was closed as already done or not worth it. Julia 1.13, `-t1`, a fresh
process after `using CausalFrames`, 1,000 rows over `readtable`:

| First call | Int time | DateTime time (run second) |
|---|---|---|
| `summarize(Mean(:x))` | 0.52 s | 0.37 s |
| `summarize([Mean, Variance, Std, Sum])` | 0.61 s | 0.46 s |
| `summarize([Mean, Std]; key = :k)` | 0.68 s | 0.20 s |
| `addrollingcolumns((w = …,), [Mean])` | 1.03 s | 0.76 s |
| `addrollingcolumns(…, [Mean, Std, Sum]; key = :k)` | 1.63 s | 0.71 s |

All of it is compilation; the DateTime calls reuse what the Int calls compiled.

- **Precompile workload.** A call shaped like a workload path (its CSV
  source, `:qty`, `summarize([Count(), Sum(:qty)]; key = :sym)` or its
  `addrollingcolumns`) still compiled 0.92 s and 1.27 s, against 0.96 s and
  1.40 s with the column renamed. The kernels are keyed by the user's schema,
  row closures and summarizer tuple, so the workload caches only the shared
  plumbing, which it already covers. Growing it buys little.
- **Widened narrow-float kernels.** `rolling.jl` and `windows.jl` test no
  Float16/Float32; and a kernel specializes on the column's element type
  anyway, so sharing state types would not share compiled code. Widening is
  #145's business, for correctness, not compile time.
- **Rejected: skip the running-table replay on a fresh `rolltiers`.**
  `rolltiers` was 1.28 s of the trace's 9.1 s (8 entries, `old::Nothing`
  each), and it compiles `replaywindow!` → `admitgroup!` even with nothing to
  replay. Constructing the empty `RunningTable` when `old === nothing` took
  `rolltiers` off the list, but `leadinsegment!` grew from 1.07 to 1.76–1.91 s:
  the lead-in kernel compiles the same admission code, which the replay had
  compiled first. The net was about −5% on the `addrollingcolumns` calls,
  within noise, so it was reverted.

What remains of a first call is the kernels (`foldgroups!`,
`leadinsegment!`, `rollsegment!`), `rowsdataframe`'s per-row-type unzip and
`summarize`'s keyed `sortedrows`, each one compile per schema and summarizer
set.

## Configuration dedup and renamed dependencies (issue #149), 2026-11-03

PR #150 deduplicates prototypes by configuration rather than by output name,
rejects two requested summarizers that share a name, and renames a clashing
hidden dependency through the `Scoped` wrapper. A set without a clash gets no
wrapper, so the question was whether the larger `prototypes` pass cost
anything. Julia 1.13.3. The two sides ran from two worktrees, master and the
branch, with identically built environments. The branch ran *master's* test
files, so only `src/` differed.

Per-file compile time from `timed_runtests.jl` (4 threads, `--depwarn=yes`),
two rounds per side, with the first side alternated:

| | master | branch |
|---|---|---|
| included files, compile | 784.2 s, 787.8 s | 788.8 s, 790.0 s |
| `rolling.jl`, compile | 251.7 s, 253.7 s | 253.7 s, 252.9 s |
| `summarizers.jl`, compile | 131.7 s, 133.5 s | 133.6 s, 132.0 s |
| `windows.jl`, compile | 90.4 s, 90.4 s | 91.0 s, 91.3 s |

The difference is +0.4%, within noise. Every file agreed to about a second.
The one outlier, `rowterms.jl` at 53.5 s in one branch round, was timed alone
four times per side, alternating, and came out at 51.8–52.4 s on master and
52.3–53.0 s on the branch.

A first-call harness ran seven fresh pipelines over 2,000 rows (`-t1`, four
rounds per side, alternating), with no clash among their summarizers.
Over the seven calls together, master took 10.28–10.89 s (mean 10.54) and
the branch 10.45–10.63 s (mean 10.55). Each call's range overlapped the other
side's, for example `addrollingcolumns` over `[Mean, Std, Skewness]` at
2.82–3.06 s against 2.77–3.04 s. The outputs matched in names, sizes and
element types. The harness data was random per process, so values were
checked by the suite instead.

The runtime A/B used `bench_run.jl` at `-t1` with four alternating rounds,
over the `summarize`, `intervalize`, `windows` and `rolling` groups plus the
controls `sources/drain` and `lastrow/keyed`. It showed a geomean of the
minimum ratios of 1.005 over 56 benchmarks. Only the control `sources/drain`
(+2.8%, identical allocations, untouched code) showed no overlap. Allocations
rose by a constant 18–45 per load, regardless of data size, and the lead-in
and chain benchmarks were unchanged. That is `prototypes`' graph bookkeeping:
for a five-summarizer set it makes 207 allocations against 138 and takes
52 μs against 49, once per pipeline built.

The rename path itself, on the branch only (master folds just one of the two
summarizers there), was measured after the harness's seven calls in the same
process, so this is the cost on top of code already compiled.
`summarize([PopSpread(), Variance(:x)])` compiled 0.30–0.35 s and the keyed
`addrollingcolumns` of the same set 1.15–1.30 s, over four runs.

## Hoisted terms placed by function (issue #151, 2026-11-04)

`prototypes` now moves a hoisted bar term apart only when two summarizers
hoist different functions under one reserved name, in place of moving a
renamed owner's own-scoped terms and matching forwarded copies by name. A
`Scoped` wrapper still appears only on such a conflict, so a call without one
compiles as before.

`timed_runtests.jl` (Julia 1.13.3, 4 threads, `--depwarn=yes`), one run per
side: included files 786.4 s compile on master, 798.0 s on the branch. The
whole difference is `barwindow.jl` (31.6 → 39.6 s), which gained the issue's
cases; every other file moved by under 3 s either way, within noise
(`rolling.jl` 242.9 → 245.5, `summarizers.jl` 146.4 → 147.7, `operators.jl`
30.8 → 29.5). The `Bars` window case runs for one summarizer set only, to bound
that cost.

Master's `barwindow.jl`, minus the five assertions whose expectations changed
(the moved name and the "unrelated" error), timed alone, alternating, two
rounds per side: 35.3–35.5 s compile on master, 35.9–36.0 s on the branch.
The extra ~0.5 s is likely the wrapper the forwarding `Relay` now gets in its
rename case, a type it didn't compile before. A run with failing assertions
reads ~4 s slower from compiling Test's failure printing, so compare only
passing files.

## PR #154 reworked: rare-path barriers, not per-row ones (2026-11-08)

PR #154 cut the suite's compile time and reported no runtime cost. Reworked
here because both claims mixed things up. Julia 1.13.3, 4 threads,
`--depwarn=yes`, `timed_runtests.jl`, compile seconds over the included
files; benchmarks at `-t1`, two worktrees with identically built
environments, 4 alternating rounds, minimums.

**Test cuts vs source.** Master 807.9; the PR's source on master's tests
679.8; the PR as opened (its trimmed tests too) 599.8. So 128 s came from
the source and 80 s from cutting test cases. Most of the cuts thinned the
`rolling.jl`/`windows.jl` differential cross products to pairwise covers,
dropping, for example, integer data on the keyless path and keyless
widening around the code the PR rewrote; those are kept here. The cuts kept
are the equivalent ones: `rowterms.jl`'s per-summarizer calls batched into
one call checking every column, the Kurtosis/Skewness edge cases batched,
and two of `keys.jl`'s three key-count cases.

**Runtime.** Against master, the PR ran 8–26% slower on every benchmark
reaching a per-row `@noinline` barrier it added: `moments/windows` 1.26,
`rolling/bars-running` 1.22, `rolling/running` 1.19, `rolling/tracking`
1.15, `moments/rolling` 1.14, `moments/running` and `windows/running` 1.12.
(The `moments` group, Kurtosis and Skewness under each transform, was added
to the A/B since nothing in `SUITE` reaches `CentralMoments`.)

**Per-site attribution.** One variant per site with only that site inlined
again, timed on master's tests, bracketed by two runs of the PR (715.8 and
682.1: identical code, 34 s apart, so single runs resolve only large
effects). `foldpowers`, `centralvalue`, `steproll!` and `admitwindow!` each
moved compile by a few seconds, within noise. Inlining `recentre!` and
`shiftpowers` (both moment states) added 27 s to `summarizers.jl` alone:
re-centring runs once per many rows, but inlined, its double-double algebra
was copied into every kernel specialization. That barrier is kept, the
per-row ones are dropped.

The per-window running tables went back from a `Vector` to an `NTuple{W}`:
with the per-row barriers gone, the `Vector` alone still cost
`rolling/bars-running` 11%, `bars-running-keyed` 9%, `running`/`tracking`
5% (V2 vs V3 below), for a compile saving within noise. The other setup
changes (unparameterized configs, the `tiering` loop, `@nospecialize` on the
construction helpers, the first-build fast path in `rolltiers` and
`preparewindows!`, projecting the chunk to `(:time, keys...)` before the
kernels, loops over comprehensions) are kept. Leave-one-out runs could only
resolve the first-build fast path (~16 s); the rest sit within the ±5%
run-to-run noise, which drifted upward through a series of runs and needed
normalizing by files the changes can't touch (`parquet.jl`, `operators.jl`,
`table.jl`).

| | master | this branch's source, master's tests | this branch |
|---|---|---|---|
| included files, compile | 807.3, 800.6 | 685.3 | 672.3 |
| `rolling.jl` | 247.0, 246.2 | 210.3 | 212.5 |
| `summarizers.jl` | 151.0, 148.4 | 102.0 | 103.6 |
| `windows.jl` | 89.7, 88.4 | 83.1 | 83.1 |
| `rowterms.jl` | 50.2, 50.0 | 38.6 | 24.8 |
| `barwindow.jl` | 37.9, 37.9 | 31.8 | 32.0 |

So the source keeps 119 s of the PR's 128 s, and the equivalent test
batching adds 13 s more (−16% in all). Runtime is at parity (next).

The `@nospecialize` `rowsdataframe` (field names read at runtime, then a
dynamic call into the generated `unzipcolumns`) went back to master's
generated form: it doubled the call, 1.3 → 2.4 µs, which read as +4.6% on
`lastrow/keyless` (one row per run), and the suite measured no compile
saving from it (686.3 normalized with it reverted, 688.1 without).

Runtime, this branch against master, minimum over 4 alternating rounds:
every `rolling`, `windows`, `summarize`, `intervalize`, `lastrow`, `join`
and `moments` benchmark within ±3% with overlapping ranges, except
`moments/summarize` at 0.92 (faster). V2 below is this branch with
`Vector` running tables, V3 with the `NTuple` it keeps:

| benchmark | PR #154 | V2 | V3 (this branch) |
|---|---|---|---|
| `rolling/running` | 1.19 | 1.09 | 1.02 |
| `rolling/bars-running` | 1.22 | 1.13 | 0.99 |
| `rolling/tracking` | 1.15 | 1.06 | 1.01 |
| `moments/windows` | 1.26 | 0.99 | 0.98 |
| `moments/running` | 1.12 | 0.99 | 0.99 |
| `windows/running` | 1.12 | 1.00 | 0.98 |
