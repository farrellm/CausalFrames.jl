# File I/O

Contents: Reading, `readcsv`, `readparquet`, `readjls`, Writing, `writecsv`, `writeparquet`, `writejls`.

Readers are sources. Writers are pass-through transforms, so they can sit
anywhere in a chain; end it with `scan` to run it for the file alone.

Parquet needs a backend, `using DuckDB` or `using Parquet2`; either enables both
operators. Reading prefers DuckDB, which pushes the window into the reader, and
writing prefers Parquet2, which streams row groups; `backend` overrides the
choice.

## Reading

Readers clip to `[start, stop)` and read incrementally.

## `readcsv`

    readcsv(path; types = nothing, time = nothing, rename = nothing,
            delim = nothing, sort = false, chunkbytes = 4 * 1024 * 1024,
            closed = false, skipmissing = false) -> CausalPipeline

A source reading the CSV file at `path`, clipped to `[start, stop)`. The file is
read incrementally, in chunks, and reading stops at the first time past the
window. Column types are **not** inferred: every column is a `String` unless
`types` says otherwise.

#### Arguments
- `path`: the file. A zero-byte file (what `writecsv` writes for an
  empty stream) reads as no rows.

#### Keywords
- `types = nothing`: concrete types for some columns, keyed by the file's own
  column names (before `rename`): a single type for every column, a vector
  indexed by column position (`nothing` entries stay `String`), a `Dict` keyed by
  name (`Symbol` or `String`) or position, or a function `(index, name) -> type`
  returning `nothing` for `String`. Unless `time` is a function, the time column
  must be given a type here.
- `time = nothing`: where `:time` comes from — `nothing` for the column named
  `:time`; a `Symbol` naming another column (after `rename`), which is renamed
  to `:time` and must not coexist with one; or a function `row -> time`,
  whose result replaces any existing `:time`. The result is converted to the
  context's time type.
- `rename = nothing`: a map (`Dict` from old to new name) or a function
  `name -> name`, applied to the column names after typing and before `time` is
  resolved.
- `delim = nothing`: the field delimiter (`Char` or `String`), passed to CSV.jl;
  `nothing` auto-detects.
- `sort = false`: stably sort the rows by time, for a file not stored in time
  order. Sorting reads the whole file and emits its in-window rows as one chunk,
  so memory scales with the window. Without it, a decreasing time is an
  `ArgumentError`.
- `chunkbytes = 4 * 1024 * 1024`: the approximate size of each chunk read.
  Must be positive.
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`.
- `skipmissing = false`: drop rows whose time is `missing` (a blank cell, or a
  `time` function returning `missing`). Without it such a row is an
  `ArgumentError`.

Order and missing-time errors are raised only for the chunks actually read.

## `readparquet`

    readparquet(path; time = nothing, rename = nothing, sort = false,
                closed = false, skipmissing = false, backend = :auto)
        -> CausalPipeline

A source reading the parquet file at `path`, clipped to `[start, stop)`. Column
types come from the file. Requires a backend: `using DuckDB` or
`using Parquet2`.

The file is read in chunks — a DuckDB result chunk, or one Parquet2 row group —
and data that cannot fall in the window is skipped undecoded: DuckDB pushes the
window into the reader, and Parquet2 skips row groups whose time statistics lie
outside it. Skipping needs the time to come from a column; with a `time`
function the file is scanned from the start, up to the first time past the
window.

#### Arguments
- `path`: the file.

#### Keywords
- `time = nothing`: where `:time` comes from, as for `readcsv`.
- `rename = nothing`: a map or a `name -> name` function applied to the column
  names before `time` is resolved, as for `readcsv`.
- `sort = false`: stably sort the rows by time, for a file not stored in time
  order. DuckDB sorts in the query and still streams; Parquet2 (and DuckDB with
  a `time` function or a `rename` hiding the time column) reads every in-window
  row and emits them as one sorted chunk. Without it, a decreasing time is an
  `ArgumentError`.
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`.
- `skipmissing = false`: drop rows whose time is `missing` (a null, or a `time`
  function returning `missing`). Without it such a row is an `ArgumentError`,
  though DuckDB's pushed-down window skips null times silently.
- `backend = :auto`: `:duckdb`, `:parquet2`, or `:auto` (DuckDB if loaded).
  Naming a backend that is not loaded is an `ArgumentError`.

Order and missing-time errors are raised only for the rows actually read.

## `readjls`

    readjls(path; closed = false) -> CausalPipeline

A source reading a file written by `writejls`, one chunk per record,
clipped to `[start, stop)`. Records are read one at a time, stopping at the
first time past the window; there is no index, so a read costs the file up to
`stop`.

#### Arguments
- `path`: the file. One not written by `writejls`, or whose last record is
  truncated, is an `ArgumentError`. Deserialization can construct any type, so
  read only files you trust.

#### Keywords
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`.

## Writing

Writers write each chunk on a background task as it flows by, and pass it on
unchanged.

## `writecsv`

    writecsv(path; queue = 1, kwargs...) -> (CausalPipeline -> CausalPipeline)
    writecsv(p::CausalPipeline, path; queue = 1, kwargs...) -> CausalPipeline

A pass-through transform writing every chunk to the CSV file at `path` as it
flows by, then yielding it downstream unchanged. Each chunk is written and
flushed by a background task, so the file grows while the pipeline runs.

#### Arguments
- `path`: the file, truncated when the pipeline starts running. An empty stream
  leaves it empty, which `readcsv` reads back as no rows.

#### Keywords
- `queue = 1`: how many chunks may wait for the writer before the pipeline
  blocks; `0` hands each chunk over directly. Must be non-negative.
- `kwargs...`: passed to `CSV.write` (`delim`, `missingstring`, `dateformat`,
  …). `append`, `header`, `writeheader`, `partition` and `compress` are
  controlled by `writecsv` and are an `ArgumentError`.

The file is complete only once the stream is exhausted — by `load`,
`scan` or a fully drained `stream`. A chunk whose column names
differ from the first chunk's is an `ArgumentError`. Use `scan` when the file
is all you want:

```julia
path = joinpath(mktempdir(), "mids.csv")
readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7])) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    writecsv(path) |>
    scan(Context(0, 10))
print(read(path, String))

# output

time,bid,ask,mid
1,10.0,10.2,10.1
2,10.5,10.7,10.6
```

## `writeparquet`

    writeparquet(path; queue = 1, rowgroupsize = 1_000_000, backend = :auto,
                 kwargs...) -> (CausalPipeline -> CausalPipeline)
    writeparquet(p::CausalPipeline, path; ...) -> CausalPipeline

A pass-through transform writing every chunk to the parquet file at `path`, then
yielding it downstream unchanged. Requires a backend: `using Parquet2` or
`using DuckDB`. Writing happens on a background task, as for
`writecsv`.

#### Arguments
- `path`: the file, truncated when the pipeline starts running.

#### Keywords
- `queue = 1`: how many chunks may wait for the writer before the pipeline
  blocks; `0` hands each chunk over directly. Must be non-negative.
- `rowgroupsize = 1_000_000`: rows per row group. Chunks are merged but never
  split, so `1` writes one row group per chunk. Must be positive. Exact under
  Parquet2; under DuckDB rounded up to a multiple of 2048.
- `backend = :auto`: `:parquet2`, `:duckdb`, or `:auto` (Parquet2 if loaded).
  Parquet2 writes each row group as it fills, holding at most `rowgroupsize`
  rows; DuckDB stages the whole output in a temporary table and writes it at
  the end. Naming a backend that is not loaded is an `ArgumentError`.
- `kwargs...`: passed to `Parquet2.FileWriter` (`compression_codec`,
  `npages`, `metadata`, …). `compute_statistics` defaults to `["time"]`, the
  statistics `readparquet` skips by. DuckDB accepts only `compression_codec`
  (`:zstd`, `:snappy`, `:gzip` or `:uncompressed`).

A parquet file is valid only once the stream is exhausted — by `load`,
`scan` or a fully drained `stream`; an abandoned stream leaves
an unusable file. An empty stream writes a valid file with no rows and only a
`time` column.

```julia
using Parquet2   # or DuckDB
dir = mktempdir()
ticks = DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7])
readtable(ticks) |> writeparquet(joinpath(dir, "ticks.parquet")) |> scan(Context(0, 10))

readparquet(joinpath(dir, "ticks.parquet")) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    writeparquet(joinpath(dir, "mids.parquet")) |>
    scan(Context(0, 10))
DataFrame(load(Context(0, 10), readparquet(joinpath(dir, "mids.parquet"))))

# output

2×4 DataFrame
 Row │ time   bid      ask      mid
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     10.0     10.2     10.1
   2 │     2     10.5     10.7     10.6
```

## `writejls`

    writejls(path; queue = 1) -> (CausalPipeline -> CausalPipeline)
    writejls(p::CausalPipeline, path; queue = 1) -> CausalPipeline

A pass-through transform writing every chunk to `path` with Julia's
`Serialization`, then yielding it downstream unchanged. It stores any value a
column can hold — fitted models, `NamedTuple`s — which CSV and parquet cannot.
Read the file back with `readjls`.

#### Arguments
- `path`: the file, truncated when the pipeline starts running.

#### Keywords
- `queue = 1`: how many chunks may wait for the writer before the pipeline
  blocks; `0` hands each chunk over directly. Must be non-negative.

Each chunk is one record, written and flushed on a background task. The file is
complete once the stream is exhausted, but every record written before an
interruption stays readable. A JLS file can be read only with a compatible
Julia and compatible versions of the packages whose types it holds.
