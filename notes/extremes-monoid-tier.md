# Why the extremes keep their deques, now that the monoid tier is a queue

This is an investigation record, not design law: `DESIGN.md` is still the source
of truth. It exists so the next person who asks "could these just be monoids?"
doesn't have to repeat the measurements.

Summary: once the monoid tier became a two-stack queue (#124/#127), it beat the
`Min`/`Max`/`First`/arg-extreme deques (`WindowTrackState`) by 0–30% at runtime
when those summarizers run alone. Moving them to the queue outright and
deleting the deques would still be a net loss. In the common mixed call (OHLC:
`[First, Max, Min, Last, Mean, Std]`) the queue saves little at runtime, and the
call then compiles two tiers' per-type machinery where it used to compile one,
which costs 9–17% of first-call compile time. Every other group loses outright
on the queue, so the running tier stays as it is.

## Environment

| | |
|---|---|
| Julia | 1.13.2, `-t1` |
| Branch | `perf/monoid-queue-v2` (#127) |
| Date | 2026-10-22 |

## Method

The tier is chosen in `tiering` (`src/tiers.jl`) by `isinvertible` of each
group's windowed state, so a scratch script forced a summarizer onto the queue
without touching the repo, by defining a method:

```julia
CausalFrames.isinvertible(::CausalFrames.WindowTrackState) = false  # the extremes only
CausalFrames.isinvertible(::CausalFrames.SummarizerState) = false   # every group (the survey)
```

The tier then folds the ordinary state (`TrackState`/`IndexState`) through
`combine!`, which is what deleting the deques would leave. Each process ran one
mode. Runtime is the minimum of 7 runs per case, over two rounds, with the modes
ordered running, queue, queue, running. Each output was hashed, and both tiers
gave identical output in all 141 cases, including a rounded series with frequent
tied extremes. So the queue's `combine!` reports the most recent extreme on a
tie, as the deque does.

## Runtime: the extremes

`addrollingcolumns` over 300,000 rows, in ns per row, including the source and
`DataFrame` overhead (about 40–50 ns per row, the cost of `Count` alone). The
data shapes were a random walk, iid noise, a rising series (which fills `Min`'s
deque to the window), a falling series (which fills `Max`'s) and a rounded walk
(ties). The windows were `Bars(5)`, `Bars(20)`, `Bars(200)`, `Bars(1000)` and
time windows of 50 and 1,000.

| call | queue / running |
|---|---|
| `Min` or `Max`, random walk | 0.68–0.83 |
| `Min` or `Max`, other shapes | 0.80–1.02 |
| `[Min, Max]` | 0.69–0.88 |
| `First`, `MaxIndex`, `MinWithIndex` | 0.81–1.11, mostly 0.85–0.96 |
| `Max` or OHLC, `summarizewindows` (`clock(10)`, look-backs 50 and 1,000) | 0.69–0.99 |
| OHLC, `addrollingcolumns` | 0.79–1.03, mostly 0.9–1.0 |
| OHLC, keyed (10 keys) | 0.90–1.09 |
| keyed `Max` at a 1,000 time window, falling series | 1.10 |

The median over all cases was 0.90. The losses are at long time windows over
deque-filling or tied data, at the edge of the round-to-round spread (up to
1.1–1.3 on some OHLC cases). A deque that fills to the window didn't hurt the
deque, since it is amortized O(1) either way; it only narrowed the queue's lead,
from about 0.75 to about 0.95. Allocations matched to within 0.3 MB per run.

## Compile time

The first harness ran five cases in one process, in a fixed order. It showed a
25–30% regression on OHLC, but that was inflated. In running mode the earlier
`Max` case had already compiled the running tier's generic machinery, while in
queue mode OHLC was the first call to need it. The harness that counts runs one
case per fresh process, after a warm-up over unrelated types (a `Float32`
column, `[Count(), Product(:z)]`, covering both tiers, `Bars` and time windows,
keyed and not, and `summarizewindows`). It reports `@timed(...).compile_time`
for 1,000 rows, two rounds per mode, with the order swapped:

| case | running | extremes on the queue |
|---|---|---|
| `Max`, `Bars(20)` | 0.87 s | 0.85–0.87 s |
| `[Min, Max]`, time window 50 | 0.89–0.95 s | 0.90–0.92 s |
| OHLC, `Bars(20)` | 1.60–1.62 s | 1.86–1.96 s |
| OHLC, keyed, time window 50 | 1.79–1.81 s | 2.09–2.14 s |
| OHLC, `Bars(5)` and time window 50 | 1.86–1.97 s | 2.18–2.28 s |
| OHLC, `summarizewindows` | 1.62–1.65 s | 1.86 s |
| OHLC, `summarizewindows`, keyed | 1.93–1.99 s | 2.10–2.16 s |

A queue-only call compiles about what a running-only call does. A mixed call
pays for both.

### Where the extra time goes

- **`--trace-compile-timing`:** the whole delta sits under the kernel,
  `rollsegment!` (0.28 → 0.61 s in the first harness). The setup code
  (`rolltiers`, `tiering`, `load`) is unchanged.
- **`@snoop_inference`, diffed by method:** 1,567 → 1,917 instances, +0.42 s of
  inference, spread thin. New are `flipqueue!`, `queuepush!`, `newqueues`,
  `livestates` and `combinenodes!`. The rest is Base's container code
  (`rehash!`, `_growend_internal!`, `unsafe_copyto!`, `_deleteat!`, `get!`)
  compiled again for the queue's own types: `Dict{K,MonoidQueues{S,R,T}}`, the
  `Vector{S}` fronts, the `Vector{PartialQueue{S}}` and the queue's own
  `Vector{R}` rows. The running tier's `RunningTable` and `RowRing` stay.
- **`Profile` with `C = true`**, after a warm-up: inference, Julia's optimizer
  and LLVM each grow about 10% (1,030 → 1,136, 270 → 296 and 427 → 466
  samples). There is no single hotspot.

### Tried and reverted

- **`@noinline` on `queuepush!` and `livestates`,** to keep the queue's code out
  of the inlined kernel: OHLC `Bars(20)` measured 1.81–1.82 s against
  1.83–1.94 s without it, within noise. The cost isn't inlining; it is a second
  set of per-type methods and containers.

### What could change the verdict

The regression is the cost of compiling two tiers for one call. Making the queue
cheaper to compile doesn't remove it, since the running tier is still there for
`Mean`/`Std`/`Last`. Only a single structure per call does.

- **All on the queue:** rejected. The survey below puts `Std` at about 1.7× and
  `Correlation` at 2.4× on it.
- **All on the running tier:** the status quo, with the deques.
- **A running-tier windowed state that is itself a two-stack queue** over the
  extremes' values: keeps OHLC single-tier. But it replaces the deque with
  another windowed state rather than deleting one, and it is unmeasured.
- **Queues that read their rows from the running tier's ring or buffer** instead
  of owning them: removes one `Vector{R}`, not the per-`S` methods.

## Survey: every group, forced onto the queue

`addrollingcolumns` over 200,000 rows of a random walk, ns per row, the
minimum of 3–7 runs. Run-to-run noise was about ±15%, so read the ratios
roughly. The windows are `Bars(20)` / `Bars(200)` / time window 50.

| summarizer | running | queue |
|---|---|---|
| `CountDistinct` | 92 / 109 / 98 | 698 / 4,589 / 1,411 |
| `SortedValues` | 120 / 171 / 200 | 225 / 639 / 303 |
| `Median` | 142 / 189 / 192 | 259 / 641 / 312 |
| `Quantile(0.9)` | 159 / 200 / 160 | 225 / 672 / 318 |
| `PercentRank` | 170 / 250 / 193 | 253 / 699 / 379 |
| `WindowValues` | 53 / 59 / 44 | 94 / 159 / 139 |
| `MeanAbsDev` | 84 / 106 / 84 | 146 / 213 / 170 |
| `Covariance` | 86 / 149 / 92 | 175 / 178 / 162 |
| `Sum` (Int) | 84 / 54 / 42 | 118 / 75 / 72 |
| `SumPower(3)` | 74 / 117 / 65 | 118 / 138 / 96 |
| `DotProduct` | 65 / 61 / 53 | 68 / 74 / 75 |
| `Moment(3)` | 95 / 77 / 80 | 118 / 100 / 101 |
| `AgeWeightedSum` | 78 / 76 / 74 | 98 / 123 / 94 |
| `Count` | 41 / 50 / 42 | 52 / 58 / 49 |
| `Last` | 57 / 53 / 40 | 65 / 60 / 64 |

From the same survey at `Bars(20)`: `Std` 104 → 176, `Correlation` 149 → 355,
`LinearRegression` 290 → 487 and `Sum` + `Mean` 130 → 159.

The growing states (sets and sorted or arrival-order vectors) pay the queue's
O(window²) front: each front slot holds a whole suffix's state. They are why
DESIGN.md's note that "no built-in pays this" depends on them staying on the
running tier. For the float sums and co-moments, the queue does remove issue
#97's residue by construction: a `Bars(5)` variance after a `1e8` spike leaves
had a worst relative error of 2.7e-16 on the queue, against 941 on the running
tier without #123's refold. But that costs 1.5–2.4× on every row, against
#123's 0–13%.
