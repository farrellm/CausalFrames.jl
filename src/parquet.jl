# Parquet I/O. Both backends are optional and either one suffices in either
# direction: the operators below hold the API, the docstrings and the
# backend-independent logic, while the machinery that names DuckDB or Parquet2
# lives in the package extensions and gives the hooks `parquetproducer` /
# `parquetsink` their real methods, one per backend.

const BACKENDS = (:duckdb, :parquet2)

const READHINT = "readparquet needs a parquet backend: run `using DuckDB` \
    (preferred — it pushes the context window into the reader) or \
    `using Parquet2`, adding it to the project if necessary"
const WRITEHINT = "writeparquet needs a parquet backend: run `using Parquet2` \
    (preferred — it writes row groups as the stream flows by) or \
    `using DuckDB`, adding it to the project if necessary"

# Given a `Val{:duckdb}` / `Val{:parquet2}` method by the extensions, so a
# missing backend can be reported where the user typed the operator.
backendloaded(::Val) = false

backendpackage(b::Symbol) = b === :duckdb ? "DuckDB" : "Parquet2"

# The backend an operator will use: the one asked for, else the preferred one if
# loaded, else the other. Called once per run — dispatching on a runtime symbol
# here costs one dynamic call per pipeline run, not per chunk — and once
# eagerly at construction, so a missing backend is reported where the operator
# was typed while one loaded afterwards still counts.
function resolvebackend(request::Symbol, preferred::Symbol, hint::String)
    if request === :auto
        backendloaded(Val(preferred)) && return Val(preferred)
        other = preferred === :duckdb ? :parquet2 : :duckdb
        backendloaded(Val(other)) && return Val(other)
        throw(ArgumentError(hint))
    end
    request in BACKENDS || throw(ArgumentError("unknown parquet backend \
        $(repr(request)); expected :auto, :duckdb or :parquet2"))
    backendloaded(Val(request)) || throw(ArgumentError("parquet backend \
        $(repr(request)) was requested but is not loaded: run \
        `using $(backendpackage(request))`"))
    return Val(request)
end

# Backend hooks. The extensions' methods are more specific than these, so the
# fallbacks only ever run when the backend is not loaded.
parquetproducer(::Val, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any) =
    throw(ArgumentError(READHINT))
parquetsink(::Val, ::Any, ::Any, ::Any, ::Any) = throw(ArgumentError(WRITEHINT))

"""
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

# Arguments
- `path`: the file.

# Keywords
- `time = nothing`: where `:time` comes from, as for [`readcsv`](@ref).
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
"""
function readparquet(path::AbstractString; time = nothing, rename = nothing,
    sort::Bool = false, closed::Bool = false, skipmissing::Bool = false,
    backend::Symbol = :auto)
    checksourcetimespec(time, "readparquet")
    resolvebackend(backend, :duckdb, READHINT)   # eager: fail at the call site
    return CausalPipeline() do ctx::Context
        return ChunkSource(
            parquetproducer(resolvebackend(backend, :duckdb,
                READHINT), ctx, String(path), time, rename, sort, closed,
                skipmissing),
        )
    end
end

"""
    writeparquet(path; queue = 1, rowgroupsize = 1_000_000, backend = :auto,
                 kwargs...) -> (CausalPipeline -> CausalPipeline)
    writeparquet(p::CausalPipeline, path; ...) -> CausalPipeline

A pass-through transform writing every chunk to the parquet file at `path`, then
yielding it downstream unchanged. Requires a backend: `using Parquet2` or
`using DuckDB`. Writing happens on a background task, as for
[`writecsv`](@ref).

# Arguments
- `path`: the file, truncated when the pipeline starts running.

# Keywords
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

A parquet file is valid only once the stream is exhausted — by [`load`](@ref),
[`scan`](@ref) or a fully drained [`stream`](@ref); an abandoned stream leaves
an unusable file. An empty stream writes a valid file with no rows and only a
`time` column.

```julia
readparquet("ticks.parquet") |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    writeparquet("mids.parquet") |>
    scan(ctx)
```
"""
function writeparquet(path::AbstractString; queue::Integer = 1,
    rowgroupsize::Integer = 1_000_000, backend::Symbol = :auto, kwargs...)
    queue >= 0 ||
        throw(ArgumentError("writeparquet queue must be non-negative, got $queue"))
    rowgroupsize >= 1 || throw(
        ArgumentError(
            "writeparquet rowgroupsize must be positive, got $rowgroupsize"),
    )
    resolvebackend(backend, :parquet2, WRITEHINT)   # eager: fail at the call site
    # Materialized once, so the per-row-group splat into the writer is over a
    # concretely typed NamedTuple rather than the keyword iterator.
    opts = values(kwargs)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            sink = parquetsink(resolvebackend(backend, :parquet2, WRITEHINT),
                String(path), Int(queue), Int(rowgroupsize), opts)
            return chunkmap(c -> sinkchunk(sink, c), p.run(ctx);
                flush = () -> finishwrite(sink))
        end
    end
end
writeparquet(p::CausalPipeline, path::AbstractString; kwargs...) =
    writeparquet(path; kwargs...)(p)

# The file's own name for the column that becomes `:time`, or nothing when it
# cannot be pinned down (a `time` function, a name that no column maps to, or
# an ambiguous `rename`). Only the window pushdown depends on this, so nothing
# is the safe answer: it costs a fuller scan, never a wrong one.
function timesourcename(filenames::Vector{String}, time, rename)
    time isa Function && return nothing
    target = time isa Symbol ? String(time) : "time"
    matches = [n for n in filenames if renamedto(n, rename) == target]
    return length(matches) == 1 ? only(matches) : nothing
end

# Column name after `rename`, mirroring what renamecolumns! does to the frame.
renamedto(n::String, ::Nothing) = n
renamedto(n::String, f) = String(f(n))
function renamedto(n::String, m::AbstractDict)
    haskey(m, n) && return String(m[n])
    haskey(m, Symbol(n)) && return String(m[Symbol(n)])
    return n
end
