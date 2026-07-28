# Specializing `SumPower`'s term for exponents 1 and 2

An investigation record, not design law — `DESIGN.md` remains the source of
truth. This note exists so the measurements behind `powerterm` in
`src/summarizers.jl` do not have to be re-derived. **Code was changed as a
result of it:** `SumPower(c, 1)` now folds `ColumnTerm{c}` and `SumPower(c, 2)`
folds `PairProductTerm{c,c}`, instead of `PowerTerm{c}` in both cases.

## The problem

The sum family shares one accumulator over a *term functor*, and
`PowerTerm{C}` carries its exponent in a **field**:

```julia
struct PowerTerm{C}
    power::Int
end
@inline termvalue(t::PowerTerm{C}, ::Type{A}, row) where {C,A} =
    convert(A, getproperty(row, C))^t.power
```

Because `t.power` is a runtime value, `^` cannot specialize on it: every row
pays a general power — `power_by_squaring` for integers, `pow_body` for floats
— where a move (`n = 1`) or a single multiply (`n = 2`) would do.

Exponents 1 and 2 are not exotic here. Exponent 2 is what every `Variance`,
`Std`, `Covariance`, `Correlation`, and `LinearRegression` depends on, whether
or not the user ever names `SumPower`; exponent 1 is what `Moment(c, 1)` uses.
`ColumnTerm{C}` and `PairProductTerm{C,C}` already compute exactly those two
quantities with the exponent in the *type*.

## Measurements

Julia 1.12, one million rows, `@belapsed`. Per-row fold cost, comparing the
four term functors over the same accumulator:

| column | `PowerTerm(1)` | `ColumnTerm` | `PowerTerm(2)` | `PairProductTerm` |
|---|---|---|---|---|
| `Int64` | 2.17 ns | **0.58 ns** | 2.08 ns | **0.65 ns** |
| `Float64` (compensated) | 5.77 ns | **2.15 ns** | 6.76 ns | **2.25 ns** |
| `Union{Missing,Float64}` (counting) | 6.75 ns | **2.32 ns** | 7.96 ns | **2.50 ns** |

So 2.7×–3.7× on the fold itself, and `downdate!` tracks it, which is what the
rolling running path pays per evicted row.

End to end over a 1M-row `Float64` column, `SumPower(:qty, 2)`:

| transform | `PowerTerm` | specialized | speedup |
|---|---|---|---|
| `summarize` | 10.5 ms | 2.8 ms | 3.77× |
| `summarizecycles` | 13.8 ms | 6.6 ms | 2.10× |
| `addsummarycolumns` | 18.1 ms | 9.9 ms | 1.83× |
| `addrollingcolumns` (window 1000) | 54.1 ms | 40.1 ms | 1.35× |
| `summarize`, keyed by symbol | 70.5 ms | 60.5 ms | 1.16× |

The ratio falls as more of the transform's own work surrounds the fold — the
keyed case is dominated by hashing a `String`-carrying key per row, which this
does not touch.

## Why it is safe

The specialization is an implementation detail: same output column name, same
accumulator type, same bits.

**Accumulator type.** `powertype(T, 1) === sumtype(T)` and
`powertype(T, 2) === dottype(T, T)` were checked for `Int8`…`Int128`, `UInt8`,
`UInt64`, `Bool`, `Float16`/`32`/`64`, `BigInt`, `BigFloat`, `Rational{Int}`,
`Complex{Float64}`, and the `Missing` unions of several of those. All agree, so
no output column's element type moves — and `widenstate` stays correct for
free, since it recomputes from `acctype(st.term, intypes)`.

**Term value.** Not bit-identical at `n = 2`, and the first version of this
note claimed otherwise. Measured against the *genuine* runtime path — `x^n`
with the exponent in a variable — over 500k random Float64 bit patterns plus
the edge-case list:

| term | vs runtime `x^n` |
|---|---|
| `n = 1` → `x` | bit-identical, everywhere |
| `n = 2` → `x*x` | **differs in 66 / 500,000** (~0.013%) |
| `Int64` (incl. `typemin`/`typemax` wrap), `Bool`, `Float32` | bit-identical |
| `±0.0`, `±Inf`, `NaN`, `floatmin`, `floatmax`, subnormals | bit-identical |

Every divergence lands in the near-underflow band (results of 4.7e-308 …
6.7e-306, against `floatmin` = 2.2e-308). Checked against 512-bit `BigFloat`,
in **all 66** cases `x*x` is the correctly rounded square and the runtime `x^2`
is the one that is 1 ULP off. So the specialization is *more* accurate — but it
is a change in output, and worth saying plainly.

What the accumulators actually depend on is unaffected. The compensated states
classify NaN and ±Inf *terms* separately and `Compensated` carries the sign of
zero through the running total, so those are the values whose bits matter — and
no nonfinite or subnormal case differs at either exponent. Only a finite term
can move, by 1 ULP, near underflow.

**One version-specific exception**, and it runs the other way. On Julia 1.10,
`^(::Float64, ::Integer)` drops the sign of zero: `(-0.0)^1` returns `0.0`.
That was fixed in 1.11. `ColumnTerm` returns the column value untouched, so on
1.10 the specialization gives `-0.0` where the old path gave `0.0` — the
specialization is the correct one. It makes no difference to the output: the
compensated accumulator's running total starts at `+0.0`, and `0.0 + -0.0` is
`0.0`, so a column of negative zeros sums to `0.0` either way (which is also
what `Sum` has always given, since it has always used `ColumnTerm`). `Float32`
is unaffected on every version. The test pins the quirk with a `VERSION` guard
rather than skipping it, so a change in either direction is noticed.

### The trap in the original check

The first version of this note rested on a test asserting `same(x^2, x*x)`,
which passes trivially and proves nothing: Julia's parser rewrites a **literal**
exponent to `Base.literal_pow(^, x, Val(2))`, and for `Float64` that *is*
`x * x`. The assertion compared `x*x` with itself. Only an exponent held in a
variable reaches `^(::Float64, ::Int)`, which is the algorithm the
specialization actually replaced.

`test/summarizers.jl` now routes every such comparison through a local
`runtimepow(x, n::Int) = x^n`, and asserts the properties above — including the
correctly-rounded-square claim and the ≤ 1 ULP bound — rather than a bit
equality that does not hold.

## What was deliberately not done

**Exponent 0.** `SumPower(c, 0)` is `Count()` in value, and `x^0` is the finite
term `1.0` even for `NaN`/`Inf` inputs — a documented behaviour the compensated
classifier depends on. Specializing it would mean a term functor that ignores
its column entirely, which buys a rarely used case a speedup it does not need
while adding a third branch to `powerterm`. Left on the general path.

**Collapsing the summarizers themselves.** This changes only which term
`SumPower` folds, *not* which summarizers deduplicate. `SumPower(:x, 1)` still
produces `:x_sumpower_1` and remains a distinct output column from `Sum(:x)`'s
`:x_sum`; `SumPower(:x, 2)` likewise stays distinct from `DotProduct(:x, :x)`'s
`:x_x_dotproduct`. Making the *names* collapse is a separate question — it is
the "diagonal representation" gap recorded under DESIGN.md's "Symmetric
summarizers", and it would change existing output column names, which this does
not.
