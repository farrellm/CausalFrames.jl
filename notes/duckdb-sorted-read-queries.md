# DuckDB sorted read: the count and quantile queries

Measured 2026-10-26 (DuckDB.jl 1.5.5, 8 cores, `JULIA_NUM_THREADS=4`) to decide
three proposed savings in `sliceboundaries`/`windowresults`. None held up; the
code is unchanged. Key-ordered files (`nkeys` runs of times `0:T-1`, columns
`time`, `key`, `x`, `s`), best of 5, ms:

| file rows | window rows | sorted read | `count(*)` | footer `num_rows` | quantile (k) | count + 1023-grid quantile | prepare |
|---|---|---|---|---|---|---|---|
| 100k | 100k (all) | 29.1 | 0.71 | 0.59 | 5.84 | 10.08 | 0.059 |
| 100k | 10k | 5.9 | 1.27 | 0.58 | 2.01 | 6.74 | 0.057 |
| 2M | 2M (all) | 418.7 | 1.21 | 0.95 | 26.72 | 35.44 | 0.060 |
| 2M | 200k | 40.2 | 3.63 | 0.79 | 7.24 | 11.61 | 0.059 |
| 20M | 20M (all) | 6000.8 | 2.14 | 1.76 | 344.07 | 331.0 | 0.066 |
| 20M | 2M | 768.2 | 87.34 | 1.58 | 115.56 | 127.75 | 0.065 |

- **Skip the count when the footer's row count is `<= sortrows`.** A
  whole-file window's count is already answered from statistics (≈ the footer
  query), so this saves at most ~0.7 ms, only on small files with a partial
  window, and costs a footer query (0.6–1.8 ms) on every larger file.
- **One count + quantile query.** The quantile must then be over a fixed grid
  (`MAXSLICES - 1` points), since `k` depends on the count. It saves the count's
  scan (~10%) on a big file whose window needs slicing but doesn't prune, and
  loses whenever the window turns out to need no slicing: issued always, a
  100k-row read pays 10 ms instead of 0.7 (+30%); gated on the footer count, a
  200k-row window of a 2M-row file pays 12.4 ms instead of 3.6 (+22%). The
  t-digest over every window row, not the count, is the cost.
- **Prepare each slice's statement once.** 0.06 ms per slice, against a slice
  query that scans the file's time column: under 0.1% even at 1024 slices.
