# Parquet I/O. Both backends are optional: the operators below hold the API,
# the docstrings and the backend-independent logic, while the machinery that
# names DuckDB or Parquet2 lives in the package extensions and gives the hooks
# `parquetproducer` / `parquetsink` their real methods.

const DUCKDBHINT = "readparquet needs the DuckDB backend: run `using DuckDB` \
    (adding it to the project if necessary) before building the pipeline"
const PARQUET2HINT = "writeparquet needs the Parquet2 backend: run \
    `using Parquet2` (adding it to the project if necessary) before building \
    the pipeline"

# Given a `Val{:duckdb}` / `Val{:parquet2}` method by the extensions, so the
# missing-backend error can be raised where the user typed the operator.
backendloaded(::Val) = false

# Backend hooks. The extensions' methods are more specific than these, so the
# fallbacks only ever run when the backend is not loaded.
parquetproducer(::Any, ::Any, ::Any, ::Any) = throw(ArgumentError(DUCKDBHINT))
parquetsink(::Any, ::Any, ::Any, ::Any) = throw(ArgumentError(PARQUET2HINT))

"""
    readparquet(path; time = nothing, rename = nothing) -> CausalPipeline

A source that reads the parquet file at `path` and clips it to the context's
half-open interval `[start, stop)`. Requires the DuckDB backend: run
`using DuckDB` first.

Parquet is self-describing, so — unlike [`readcsv`](@ref) — there is nothing to
type by hand. The file is read in DuckDB-sized chunks, never all at once, and
the context window is pushed down into the reader, so row groups and pages
outside `[start, stop)` are never decoded.

The resulting time column, whatever its source, is materialized as `:time`,
must be sorted in non-decreasing order, and is converted to the context's time
type. It is chosen by `time`:

- `time = nothing` (default): the column already named `:time`.
- `time = :name` (a `Symbol`): the column named `:name` (after `rename`),
  renamed to `:time`.
- `time = f` (a function): `f(row)` is called per row to compute the time
  value, producing the `:time` column (any existing `:time` is overwritten).

Keyword arguments:

- `rename`: an `AbstractDict`/map (over the file's own names) or a
  `name -> name` function applied to the column names **before** `time` is
  resolved.

The window is pushed down only when the time values come from a real column: a
`time` function is opaque to the reader, so such a file is scanned from the
start (it still stops as soon as a time `>= stop` is seen). Pushdown never
changes results — the rows are clipped again on arrival — so a file whose
writer recorded no statistics simply reads more of itself. Consequently, as
with [`readcsv`](@ref), a sortedness violation is only detected in the chunks
actually read.
"""
function readparquet(path::AbstractString; time = nothing, rename = nothing)
    backendloaded(Val(:duckdb)) || throw(ArgumentError(DUCKDBHINT))
    return CausalPipeline() do ctx::Context
        return ChunkSource(parquetproducer(ctx, String(path), time, rename))
    end
end

"""
    writeparquet(path; queue = 1, rowgroupsize = 1_000_000, kwargs...) ->
        (CausalPipeline -> CausalPipeline)
    writeparquet(p::CausalPipeline, path; ...) -> CausalPipeline

A transparent pass-through transform that writes the stream to the parquet file
at `path` as it flows by, yielding every chunk downstream unchanged. Requires
the Parquet2 backend: run `using Parquet2` first.

Like [`writecsv`](@ref), writing happens on a background task fed by a bounded
queue, so the pipeline does not block on disk I/O — only if the writer falls
more than `queue` chunks behind, plus once at the end to join it. Chunks are
accumulated until `rowgroupsize` rows are pending and then written as one row
group; chunks are only ever merged, never split, so `rowgroupsize = 1` writes
one row group per incoming chunk.

Unlike a CSV file, **a parquet file is only valid once finalized**, which
happens when the stream is *exhausted* — by [`load`](@ref), [`scan`](@ref), or
a fully drained [`stream`](@ref). There is no usable prefix on disk while the
run is in flight, and abandoning a `stream` part-way leaves an unusable file;
use [`scan`](@ref) when the file is all you want:

```julia
scan(ctx, readparquet("ticks.parquet") |>
          addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
          writeparquet("mids.parquet"))
```

A stream with no rows at all yields a valid file with no columns. Keyword
arguments are passed through to `Parquet2.FileWriter` (`compression_codec`,
`npages`, `metadata`, `column_metadata`, …); `compute_statistics` defaults to
`["time"]`, so files written here carry the statistics [`readparquet`](@ref)'s
pushdown reads.

The curried form composes with `|>`; the uncurried form applies directly, so
`writeparquet(p, path)` is equivalent to `p |> writeparquet(path)`.
"""
function writeparquet(path::AbstractString; queue::Integer = 1,
    rowgroupsize::Integer = 1_000_000, kwargs...)
    backendloaded(Val(:parquet2)) || throw(ArgumentError(PARQUET2HINT))
    queue >= 0 ||
        throw(ArgumentError("writeparquet queue must be non-negative, got $queue"))
    rowgroupsize >= 1 || throw(
        ArgumentError(
            "writeparquet rowgroupsize must be positive, got $rowgroupsize"),
    )
    # Materialized once, so the per-row-group splat into the writer is over a
    # concretely typed NamedTuple rather than the keyword iterator.
    opts = values(kwargs)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            sink = parquetsink(String(path), Int(queue), Int(rowgroupsize), opts)
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
