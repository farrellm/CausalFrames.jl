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
parquetproducer(::Val, ::Any, ::Any, ::Any, ::Any) =
    throw(ArgumentError(READHINT))
parquetsink(::Val, ::Any, ::Any, ::Any, ::Any) = throw(ArgumentError(WRITEHINT))

"""
    readparquet(path; time = nothing, rename = nothing, backend = :auto) ->
        CausalPipeline

A source that reads the parquet file at `path` and clips it to the context's
half-open interval `[start, stop)`. Needs one of the two parquet backends
loaded — `using DuckDB` or `using Parquet2` — and uses DuckDB when both are.

Parquet is self-describing, so — unlike [`readcsv`](@ref) — there is nothing to
type by hand. The file is read in chunks, never all at once, and the context
window is used to skip data that cannot be in it:

| `backend` | Chunk | Window |
|---|---|---|
| `:duckdb` (preferred) | a DuckDB result chunk | pushed into the reader: row groups *and* pages outside the window are never decoded |
| `:parquet2` | one row group | row groups whose recorded time statistics fall outside the window are skipped undecoded |

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
- `backend`: `:auto` (default), `:duckdb` or `:parquet2`. Naming a backend that
  is not loaded is an `ArgumentError`.

Skipping applies only when the time values come from a real column: a `time`
function is opaque to the reader, so such a file is scanned from the start (it
still stops as soon as a time `>= stop` is seen). It never changes results —
the rows are clipped again on arrival — so a file whose writer recorded no
statistics simply reads more of itself. Consequently, as with [`readcsv`](@ref),
a sortedness violation is only detected in the chunks actually read.
"""
function readparquet(path::AbstractString; time = nothing, rename = nothing,
    backend::Symbol = :auto)
    resolvebackend(backend, :duckdb, READHINT)   # eager: fail at the call site
    return CausalPipeline() do ctx::Context
        return ChunkSource(
            parquetproducer(resolvebackend(backend, :duckdb,
                    READHINT), ctx, String(path), time, rename),
        )
    end
end

"""
    writeparquet(path; queue = 1, rowgroupsize = 1_000_000, backend = :auto,
                 kwargs...) -> (CausalPipeline -> CausalPipeline)
    writeparquet(p::CausalPipeline, path; ...) -> CausalPipeline

A transparent pass-through transform that writes the stream to the parquet file
at `path` as it flows by, yielding every chunk downstream unchanged. Needs one
of the two parquet backends loaded — `using Parquet2` or `using DuckDB` — and
uses Parquet2 when both are.

Like [`writecsv`](@ref), writing happens on a background task fed by a bounded
queue, so the pipeline does not block on disk I/O — only if the writer falls
more than `queue` chunks behind, plus once at the end to join it. Chunks are
only ever merged, never split, so `rowgroupsize = 1` writes one row group per
incoming chunk. How they reach the file depends on the backend:

| `backend` | Writing | Memory |
|---|---|---|
| `:parquet2` (preferred) | one row group per `rowgroupsize` buffered rows, written as the stream flows by | bounded by `rowgroupsize` |
| `:duckdb` | chunks are staged in a temporary DuckDB table and written by a single `COPY` when the stream ends | scales with the whole output (DuckDB spills to its temp directory) |

`rowgroupsize` is exact under Parquet2 and a hint under DuckDB, which rounds it
up to a multiple of its own 2048-row vector size.

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

A stream with no rows at all yields a valid file with no rows.

`backend` is `:auto` (default), `:parquet2` or `:duckdb`; naming one that is not
loaded is an `ArgumentError`. Remaining keyword arguments are passed through to
`Parquet2.FileWriter` (`compression_codec`, `npages`, `metadata`,
`column_metadata`, …), where `compute_statistics` defaults to `["time"]` so that
files written here carry the statistics [`readparquet`](@ref) skips by. The
DuckDB backend understands `compression_codec` (`:zstd`, `:snappy`, `:gzip`,
`:uncompressed`) and records statistics of its own, but rejects the other,
Parquet2-specific options.

The curried form composes with `|>`; the uncurried form applies directly, so
`writeparquet(p, path)` is equivalent to `p |> writeparquet(path)`.
"""
function writeparquet(path::AbstractString; queue::Integer = 1,
    rowgroupsize::Integer = 1_000_000, backend::Symbol = :auto, kwargs...)
    resolvebackend(backend, :parquet2, WRITEHINT)   # eager: fail at the call site
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
