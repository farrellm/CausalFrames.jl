# Variance cancellation under a large offset (issue #89)

An investigation record, not design law. DESIGN.md's "Dependent summarizers"
section describes the design that resulted.

## The failure

`Variance` used to be `(Σx² − (Σx)²/n)/(n − corrected)` over the shared
`Count`, `Sum` and `SumPower(x, 2)` accumulators. Under a window, at a level
large against the spread, it lost most of its digits. `barwindow` with
`corrected = false`, 400 uniform values of spread ~577 plus an offset `c`,
worst relative error of `var(x + c)` against `var(x)`:

| window | c = 1e6 | c = 1e8 | c = 1e10 |
|---|---|---|---|
| `Bars(2)` | 1.6e-2 | 5.2e2 | 2.1e4 |
| `Bars(5)` | 2.9e-8 | 1.2e-4 | 3.8 |
| `Bars(20)` | 1.2e-9 | 8.9e-6 | 0.12 |

A 1e8 level with 0.01 steps gave a variance of `-2.0` (true value ~1e-4).

## Why better summation cannot fix it

The sums were already compensated (Neumaier). The digits are lost before the
summation and after it:

1. each term `x * x` is rounded at the scale of `x²` (about 2 at x = 1e8);
2. the dependent reads `Σx²` as one rounded float, so even an exact
   (two-product, double-double) sum is cut to `eps · Σx²` before the
   subtraction.

So the fix had to centre *before squaring*, in a state the variance reads
directly.

## Alternatives rejected

- **Welford/Chan** (mean and M2, with the inverse update for `downdate!` and
  the parallel formula for `combine!`): a division per row, and the inverse
  update is uncompensated, so its error random-walks over a long running
  window and persists after a volatile stretch leaves the window, the case the
  compensated sums exist for. A NaN or ±Inf poisons the mean and cannot be
  removed without the same counters the shifted state uses anyway.
- **Error-free squares** (`fma` two-product into the compensated sums) and a
  double-double final formula, keeping the shared raw sums: dependents would
  need the compensated *pairs*, changing the value interface of `Sum`,
  `SumPower` and `DotProduct`, and `(Σx)²/n` would need double-double too.
  Roughly twice the per-row work and far more invasive.
- **A shift inside `Sum`'s own state**: `Sum` must report the true `Σx`, and
  it is shared with `Mean`, regressions and direct requests.
- **Clamping alone**: hides the negative sign but not the error. Kept only as
  a final guard (`max(c, 0)` when `a = b`).
- **A relative floor** (the issue's "below `1e-12 · mean(x²)` is 0", after
  TA-Lib): with centred sums the residue is at the scale of the spread, and a
  floor relative to the raw level would zero genuinely small variances at high
  levels.

## The shifted co-moment, measured

The unexported `CoMoment{a,b}` folds compensated `Σ(a − Ka)`, `Σ(b − Kb)` and
`Σ(a − Ka)(b − Kb)`, and recentres once more than half its rows arrived since
the shift last moved. Same data, worst relative error against an exact
(`BigFloat`) variance of the *same rounded inputs*, so representation error is
excluded:

| window | c = 0 | c = 1e6 | c = 1e8 | c = 1e10 |
|---|---|---|---|---|
| `Bars(2)` | 9.7e-8 | 6.6e-7 | 2.5e-7 | 2.0e-7 |
| `Bars(5)` | 3.3e-13 | 1.1e-13 | 8.9e-14 | 1.1e-13 |
| `Bars(20)` | 5.4e-15 | 1.5e-15 | 3.1e-15 | 1.8e-15 |

The error no longer depends on the offset. The `Bars(2)` residue is the
two-row window whose spread is tiny against its distance from the shift (two
near-equal values after a large jump): the shift is chosen before the second
row arrives, so `eps · (distance / spread)²` is lost whatever the level,
`c = 0` included. The 1e8/0.01 series now has minimum variance 2.5e-5 (true
values ~1e-4 and up), a flat 1e-6 window gives exactly 0, and a 200,000-row
`Bars(5)` over a trend from 1e5 upward stays within 2.7e-12 of exact (it was
2.4e-4), which is what the recentring buys.

### Recentring target: the newest row, not the mean

Recentring to the current means was 3-10x more accurate on this random data
(`Bars(5)` 2.6e-14, `Bars(20)` 4.4e-16), but `S1/n` is generally inexact, so
the deviations stop being exact differences of inputs. The running tier and the
re-fold oracle then disagreed on windows whose true co-moment is exactly 0
(1e-17 against 0, which `isapprox` rejects): the rolling and windows
differential tests failed on quarter-step float data. Recentring to the row
just folded keeps the deviations exact differences of data values, about one
spread from the mean, so states holding the same rows agree exactly wherever
the inputs allow. The value is formed as `(n·Σab − Σa·Σb)/n` for the same
reason: one rounding, with an exact numerator for integers.

A single folded row returns exactly 0 rather than reading the sums: after
sliding, compensated `Σd²` need not equal `d·d` bit for bit, and a corrected
one-row variance must be `0/0 = NaN`, not `±Inf`.

### The `Bars(2)` floor, and why recentring more often doesn't lower it

(Superseded by "Error-free products" below, which removed this floor.)

CausalIndicators' port of TA-Lib's `test_stddev.c` checks shift invariance
to `1e-9 + 4c·eps/σ`. With this design it passes in every cell except
`Bars(2)` at `c = 1e6`, where the tolerance is tightest and the c-independent
floor above (pairs of near-equal values far from the shift) exceeds it. Its
non-negativity leg passes.

Recentring once a *quarter* of the rows postdate the shift (so `Bars(2)`
shifts to the newest row on every row) made it worse: `Bars(2)` 1.1e-6 and
`Bars(5)` 7.1e-13 worst. Moving the shift by δ rewrites the sums with error
~`eps·n·δ²`, at the scale of the jump, so more frequent moves add error rather
than remove it. Lowering the floor needs the deviations' squares and the
rewrite carried exactly: error-free products (`fma`) into the compensated sums
and a double-double evaluation in `value`, all local to `CoMomentState`. That
was left as a possible follow-up; it would cost roughly another fma and
compensated step per product per row.

## Error-free products (issue #91)

The floor above was then removed by the follow-up it names, kept local to
`CoMomentState`: each `d_a·d_b` is an `fma` two-product whose error goes
straight into the Neumaier compensation (so `downdate!` removes the identical
pair), the recentring and `combine!` rewrites form `δ·Σd` against the
compensated pair's total exactly and `n·δa·δb` as two two-products, and
`value` evaluates `n·Σab − Σa·Σb` over the pairs `total + comp` in
double-double, rounding once. Integer and `BigFloat` states keep plain
arithmetic. With exact inputs every low part is zero, so the formula reduces
to the old one and exact agreement on a zero co-moment is unchanged.

The #90 table again (same data, same exact reference):

| window | c = 0 | c = 1e6 | c = 1e8 | c = 1e10 |
|---|---|---|---|---|
| `Bars(2)` | 8.3e-12 | 0 | 0 | 0 |
| `Bars(5)` | 4.3e-15 | 3.4e-16 | 3.3e-16 | 3.3e-16 |
| `Bars(20)` | 4.6e-16 | 3.4e-16 | 3.0e-16 | 2.8e-16 |

The issue's LCG series (`100 + 20·u`, TA-Lib's `test_stddev.c`) is now exact
at `Bars(2)` on every row (was 5e-7), and `var(c·x)` matches `c²·var(x)` to
8e-12 for `c ∈ {3, 1e3, 1e-3}`. The 20,000-row drift at `Bars(5)` is within
3.5e-16 of exact.

The remaining `c = 0` error is the deviation `x − K`: one subtraction, exact
when `x` and `K` are within a factor of two (Sterbenz), else rounded at
`eps·D`, costing `eps·D/s`. The test data there crosses zero, so x and K
can't be within 2× of each other. An exact `TwoDiff` deviation would remove
it, but then the recentring shift `δ` has to be carried as a pair too, which
makes every rewrite term double-double. It was left as is: a linear `eps·D/s`
only matters for a spread many orders below its distance to the previous level.

The issue's own suggestion, moving `K` to the newest row (or the window mean)
on every eviction, was not taken: the quarter-threshold measurement above
already showed that with an inexact rewrite it adds error, and with an exact
one it buys nothing, since the deviations no longer round at `D²`.

Cost, 200,000 rows, single-threaded, ns/row, mean of two interleaved runs
(run-to-run noise here is about ±10%):

| case | before | error-free |
|---|---|---|
| `Std`, `Bars(20)` | 121 | 122 |
| `Std`, time window 50 | 97 | 91 |
| `Correlation`, `Bars(20)` | 136 | 150 |
| `LinearRegression` K = 1, `Bars(20)` | 204 | 231 |
| `LinearRegression` K = 3, `Bars(20)` | 1203 | 1236 |
| `summarize(Variance)` | 14.5 | 16.3 |

## Cost

200,000 rows, single-threaded, ns/row (master → this design):

| case | master | co-moment |
|---|---|---|
| `Std`, `Bars(20)` | 109 | 94 |
| `Std`, time window 50 | 74 | 84 |
| `Mean` + `Std`, `Bars(20)` | 88 | 101 |
| `Correlation`, `Bars(20)` | 111 | 133 |
| `LinearRegression` K = 1, `Bars(20)` | 164 | 195 |
| `LinearRegression` K = 3, `Bars(20)` | 951 | 1195 |
| `summarize(Variance)` | 13 | 19 |
| `barwindow` `Std(20)` | 24 | 31 |

An off-diagonal co-moment folds three sums where a dot product folded one, and
a `Mean` beside a `Variance` no longer shares the variance's sum. The
alternative for regressions, one multivariate co-moment accumulator per fit
(one shift per column, as many sums as before), would recover most of the
regression overhead but lose the sharing with `Variance`/`Covariance`; at
20-26% it wasn't judged worth a second state type.

## Refolding after a dominant row leaves (issue #97)

Under a window, a row much larger than the rest left a residue once it had
gone. CausalIndicators' port of TA-Lib's `test_reference.c` found it: a `1e8`
print among prices near 100 kept a `Bars(5)` variance 500-600 times too large
for the rest of the stream (300 bars and counting), pandas GH#52407's tiny
values under `Bars(3)` read `3.3e-48` for `1e-102` to `6.5e-96`, and pandas'
`±3e37` case gave a `NaN` correlation after both extremes had left. Time
windows, `summarizewindows` and `barwindow` showed it too; the tree and
refold tiers never did, since they don't downdate.

### Where the residue comes from

Stepping a `CoMomentState` through the spike case: while the spike is in the
window the value is exact to 1e-16. The shift then moves (rows 102 and 105),
and when the spike leaves at 106, `Σd` is off by 3.5e-8 and `Σd²` by 2.15,
against a true `Σd²` of 0.005. Two mechanisms, either enough alone:

1. **A row's deviation is rounded at its own scale and removed under another
   shift.** `1e8 - K` rounds at about `eps·1e8 ≈ 1.5e-8`. Folded under one
   `K` and removed under a later one, it rounds differently, and the
   difference times `2·d ≈ 2e8` (about 2) stays in `Σd²`. The recentring
   rewrite is exact, so it carries the rounding along unchanged. This is the
   `eps·D/s` cost the error-free-products section above left in place, at a
   `D` it didn't anticipate: a single row, not a level.
2. **Compensation can't hold what is far below it.** The Neumaier `comp` is a
   plain float. Once it holds a `1e16`-scale term's rounding, anything about
   1e-16 below that is lost. This alone gives the GH#52407 and `±3e37`
   failures, and it also reaches the plain sums: on GH#52407, `Sum` keeps a
   lasting 4e-4 relative error and `SumPower(:x, 2)` 2e-7 (fixed the same
   way; see "The sums" below).

### The fix: ask to be refolded

Only a refold from the live rows removes both, since sliding has already lost
the information. The design is in DESIGN.md's "Dependent summarizers": a
float state keeps `peak`, the largest `|d_a·d_b|` folded since its sums
restarted, and asks after a `downdate!` once `peak > 2^10 · scale`, with the
scale `Σd²` for a variance and `√(Σd_a²·Σd_b²)` for a covariance. The two
plain sums are kept for the covariance alone, since a covariance near zero is
legitimate and comparing against it would refold on every eviction. A live
row's product is at most the scale, so the state asks only once a dominant
row has left, and the residue that stays short of the trigger is at most
about `eps · 2^10` of the scale (about 2e-13).

After the fix every path is exact to 2e-16 on the spike (master: 607 on
`Bars(5)`, 375 on a time window, 607 on `summarizewindows` and `barwindow`).
GH#52407 and the `±3e37` correlation match the exact values. A sweep shaped
like CausalIndicators' `test_beta.c` outlier transit (spikes of 1e3 to 1e12,
1e-3 and 1e-9, on either column, at periods 5, 14 and 30) has a worst error of
2.2e-16 of the scale after the spike leaves, against 6.5e13 on master.

How often it asks, per 10⁶ rows of `refoldrate.jl`'s data:

| data | `Bars(5)` var / cov | `Bars(20)` | `Bars(100)` |
|---|---|---|---|
| random-walk levels | 35 / 0 | 0 / 0 | 0 / 0 |
| its steps | 5 / 0 | 0 / 0 | 0 / 0 |
| Cauchy steps | 14,234 / 7,245 | 2,811 / 1,365 | 554 / 253 |

The Cauchy refolds are warranted: each follows the departure of a product
more than `2^10` times what the window still holds, whose rounding would
otherwise stay. Even there the extra work is about
0.06-0.07 row folds per row.

### Alternatives rejected

- **Exact deviations** (`TwoDiff` for `x − K`, the shift carried as a pair):
  removes mechanism 1 but not 2, at a cost on every row.
- **TA-Lib's trigger**, comparing the leaving term to what remains: misses a
  slow decline, where no single leaving term is large against the rest but
  the residue of all of them is. A high-water mark sees it. TA-Lib's
  fixed-interval rebuild is unnecessary with the mark.
- **A ratio near 1**: ordinary data at short windows would refold often. At
  `2^10` the bound is about 2e-13 of the scale, and ordinary data rarely
  asks.
- **Refolding only the asking states** (a generated fold over the states that
  can ask, the others kept): far worse to compile. The keyed `leadinsegment!`
  went from 387 ms to 3,056 ms and `windowrows!` from 471 to 2,390 ms, the
  compile harness from 9.2 to 15.7-22 s. Refolding the whole group through
  `freshall!`/`updateall!`, which admission has already compiled, brought the
  harness back to 9.5-9.6 s.
- **An unspecialized refold** (dispatched per row, compiled only if a refold
  happens): would cost no compile time up front, but `test/jet.jl`'s tier sets
  include `Std`, so `@test_opt` would report the dispatch.

### Cost

Runtime, 200,000 rows of a correlated random walk, single-threaded, ns/row,
minimum over four alternating rounds of two identically built worktrees
(master and the branch). Allocations are unchanged in every case.

| case | master | refold |
|---|---|---|
| `Sum`, `Bars(20)` (control) | 40.6 | 40.7 |
| `Std`, `Bars(20)` | 72.5 | 70.6 |
| `Std`, `Bars(20)`, 10 keys | 147.7 | 149.0 |
| `Std`, time window 50 | 64.8 | 67.4 |
| `Covariance`, `Bars(20)` | 73.7 | 76.6 |
| `Correlation`, `Bars(20)` | 127.6 | 133.2 |
| `LinearRegression` K = 1, `Bars(20)` | 179.7 | 184.9 |
| `summarizewindows`, `Std` + `Correlation` | 75.8 | 80.4 |
| `barwindow` `Std(20)` | 40.7 | 41.3 |

Compile time: a harness of seven first calls with co-moments (rolling `Bars`
and time windows, keyed, a regression, `summarizewindows`, `barwindow`) went
from 9.05-9.24 s to 9.39-9.55 s over four runs each. The timed test suite's
included files compiled in 641.8 s on master and 649.1 s on the branch, the
difference being this issue's own new testset (about 10 s); `rolling.jl` and
`windows.jl` were unchanged within noise.

### The sums

The sum family (`Sum`, `SumPower`, `DotProduct`, and `Mean` through `Sum`) has
only the second mechanism, having no shift. Its windowed state stores a
`Tracked` (the `Compensated` plus `peak`, the largest finite `|term|`, and
`mag`, a plain live `Σ|term|`) and asks once `peak > 2^10 · mag`. `Σ|term|`
rather than the sum itself, which may cancel to near zero: against `|Σ|`,
alternating data would refold on every eviction.

On GH#52407 under `Bars(3)`, the windows that read `0` for `2.2e-51` (relative
error 1.0) on the previous commit are now exact for `Sum`, `SumPower(x, 2)`,
`SumPower(x, 3)` and `DotProduct`, on rolling, keyed time windows,
`summarizewindows` and `barwindow`.

Asks per 10⁶ rows (the same data as the co-moment table):

| data | `Sum` / `SumPower(x, 2)` at `Bars(5)` | `Bars(20)` | `Bars(100)` |
|---|---|---|---|
| random-walk levels and steps | 0 / 0 | 0 / 0 | 0 / 0 |
| Cauchy steps | 617 / 13,924 | 46 / 3,051 | 3 / 595 |

**Only the windowed state tracks.** The first version tracked in every
`AccumState`. On the folds that never downdate it cost `summarize/keyless`
24%, `summarize/cycles` and `summarize/running` 11-12% and
`summarize/powersum` 8% (`ab_bench.sh`, against the co-moment commit), so
the tracking moved to the `freshwindowed` storage, and the co-moment's to its
windowed state (a flag `W`) for the same reason: `powersum`'s 8% was partly
its `Variance`. Afterwards, against master, `summarize/keyless` read 1.010 and
`powersum` 1.003.

The windowed sums still pay. In a scratch benchmark of rolling `Sum` + `Mean`
over 400k rows, with matched environments, ns/row:

| case | master | branch | branch, no check |
|---|---|---|---|
| `Bars(100)` | 48.7-51.2 | 52.7-53.3 | 50.9-51.8 |
| time window 25 | 42.9-44.8 | 46.0-46.7 | 45.8-45.9 |
| `summarizewindows` | 38.7-39.7 | 40.8-41.6 | 40.0-40.5 |

That's 3-5%, about half of it the per-eviction check. `ab_bench.sh` read
8-14% on the `SUITE` window-sum leaves, but it carries the ~5% bias recorded
in `notes/compile-time.md` (repo against worktree environments). Folding the
tracking into `compadd`'s finite branch, to test `isfinite` once, measured
within noise and was not kept.

Compile time: splitting the windowed states from the ordinary ones means a
window call compiles both types where they used to be one. The co-moment
harness went from +3% (tracking everywhere) to +4-5% over master (9.73-9.98 s
against 9.12-9.47 s).
