# DuckDB: why the pushed-down window does not surface null times

`readparquet(; skipmissing = false)` makes a missing time an error, but only in
the rows actually read. Under DuckDB the context window is pushed down as
`WHERE col >= ? AND col < ?`, which also drops NULL times, so a null in a named
time column goes unreported there, while Parquet2 (which clips every row group
it reads) raises it. Results that succeed agree between backends. The error
coverage differs, as it already did for sortedness: disorder outside the window
but inside an overlapping row group also errors under Parquet2 and not under
DuckDB.

The obvious fix, `WHERE (col >= ? AND col < ?) OR col IS NULL`, was measured
and rejected (DuckDB 1.5.5, Julia DuckDB.jl, 10M rows, 100 row groups of 100k,
a 1k-row window, warm):

| file | plain window | `OR col IS NULL` |
|---|---|---|
| a NULL every 1000 rows | 9.7 ms | 90.3 ms |
| nullable column, no NULLs | 5.1 ms | 36.2 ms |

The disjunction defeats the row-group skip even when the file's statistics show
no nulls, so the clause would slow every default DuckDB read ~7x to report a
condition most files never have.
