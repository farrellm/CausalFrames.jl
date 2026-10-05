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
