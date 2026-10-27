# The DuckDB backend, the only code that names DuckDB. Reading (preferred) is
# one streaming query per run, pulled a result chunk at a time, with the window
# pushed down as a WHERE clause so DuckDB skips row groups and pages outside it.
# Writing (the fallback) stages the stream in a temporary table and writes the
# file with one COPY, since DuckDB can't append row groups to a parquet file.
module CausalFramesDuckDBExt

using CausalFrames
using DataFrames
using DuckDB
using Tables

using CausalFrames:
    ChunkSink, PullCursor, SourceClip, pullclipped!, keptfilecolumns, timesourcename

CausalFrames.backendloaded(::Val{:duckdb}) = true

# One in-memory database per process, created on first use (never at
# precompile time), and a connection per pipeline run, since a connection is
# single-consumer and a pipeline may run more than once (a self-join reads its
# file twice). A connection costs microseconds, the database milliseconds.
const DBLOCK = ReentrantLock()
const DB = Ref{Any}(nothing)

function connection()
    return lock(DBLOCK) do
        DB[] === nothing && (DB[] = DuckDB.DB())
        return DBInterface.connect(DB[])
    end
end

# The stateful producer behind readparquet's ChunkSource. The untyped fields
# are per-chunk setup; per-row work is `clipchunk!`'s.
mutable struct QueryProducer{T}
    const clip::SourceClip{T}
    const sort::Bool
    const sortrows::Int # rows a sorted read sorts at a time, roughly
    gather::Bool        # sort in Julia, the query being unable to
    con::Any            # DuckDB connection, opened on first pull
    parts::Any          # PullCursor over the result chunks, as DataFrames
end

CausalFrames.parquetproducer(::Val{:duckdb}, clip::SourceClip{T},
    sort::Bool, sortrows::Int) where {T} =
    QueryProducer{T}(clip, sort, sortrows, false, nothing, nothing)

function (p::QueryProducer)()
    p.parts === nothing && startquery!(p)
    return pullclipped!(p.clip, p.parts, p.gather)
end

# Open the connection and start the streaming scan. The window is pushed down
# whenever the time column can be named in SQL; chunks are clipped on arrival
# regardless, so a failed pushdown only costs a longer read.
#
# A `sort` is pushed down too, as an ORDER BY on the time column. DuckDB's sort
# is unstable, so ties are broken by `file_row_number`, the reader's virtual
# column (not in `*`), unless a file column of that name (case-insensitively)
# shadows it. Without a nameable time column, a usable tiebreak or a window
# DuckDB can bind, the rows are sorted in Julia.
function startquery!(p::QueryProducer)
    clip = p.clip
    p.con = connection()
    filenames = columnnames(p.con, clip.path)
    src = timesourcename(filenames, clip.time, clip.rename)
    p.gather =
        p.sort &&
        (src === nothing || any(n -> lowercase(n) == "file_row_number", filenames))
    sel = selectlist(filenames, clip)
    results = nothing
    if src !== nothing
        try
            results = windowresults(p, quotename(src), sel)
        catch e
            # A time type DuckDB can't bind or compare with this column: read
            # the whole file and let the clip do the work, gathering under a
            # sort so DuckDB never sorts (and holds) the whole file. An
            # unreadable file fails again below with its own error.
            isduckdberror(e) || rethrow()
            p.gather = p.sort
        end
    end
    results === nothing && (results =
        (execstream(p.con, "SELECT $sel FROM read_parquet(?)", Any[clip.path]),))
    p.parts = PullCursor(DataFrame(t) for r in results for t in Tables.partitions(r))
    return nothing
end

# The streaming results of the window's query, in order. A sorted read is cut
# into time slices of about `sortrows` rows, each its own `ORDER BY` query, so
# DuckDB holds one slice's rows at a time rather than the window's; any other
# read is one slice. Ties never straddle a half-open slice boundary, so the
# order stays stable. A window DuckDB can't bind fails here, in the first query
# to name it: a sorted read's count, so no sort runs inside the caller's bind
# fallback, or else the one streaming query. The slices share a prepared
# statement (a second for the last when `closed`), and run lazily, each once
# the previous result is drained.
function windowresults(p::QueryProducer, col::String, sel::String)
    clip = p.clip
    sorted = p.sort && !p.gather
    bounds = sorted ? sliceboundaries(p.con, clip, col, p.sortrows) : Any[]
    lows, highs = Any[clip.start; bounds], Any[bounds; clip.stop]
    order = sorted ? " ORDER BY $col, file_row_number" : ""
    query(closed) = DBInterface.prepare(p.con,
        "SELECT $sel FROM read_parquet(?) $(windowfilter(col, closed))$order",
        DuckDB.StreamResult)
    final = query(clip.closed)
    inner = clip.closed && length(lows) > 1 ? query(false) : final
    slice(i) = DBInterface.execute(i == length(lows) ? final : inner,
        Any[clip.path, lows[i], highs[i]])
    return sorted ? (slice(i) for i in eachindex(lows)) : (slice(1),)
end

# The errors DuckDB raises for a query it can't bind or run; anything else (an
# interrupt, a bug of ours) is no reason to retry the read another way.
isduckdberror(e) = e isa Union{DuckDB.QueryException,DuckDB.NotImplementedException}

# The WHERE bounding the time column by two parameters, `[lo, hi)` or, closed,
# `[lo, hi]`. It also drops null times, so without `skipmissing` they go
# unreported here; `OR $col IS NULL` would cost the row-group skip (see
# notes/duckdb-null-pushdown.md).
windowfilter(
    col::String,
    closed::Bool,
) = "WHERE $col >= ? AND $col $(closed ? "<=" : "<") ?"

# The times splitting the window into slices of about `sortrows` rows, at most
# `MAXSLICES` of them: none when it holds no more, else DuckDB's approximate
# quantiles, in the context's time type and strictly inside the window. A
# window DuckDB can't bind fails the count. Quantiles DuckDB can't take, or
# that collapse once converted and clipped (ties, or a floating column floored
# into an integer context), can leave far fewer slices than asked for, each a
# sort DuckDB holds whole: under half of them warns, judged on the cuts that
# remain, since a cut at or outside the window bounds nothing.
function sliceboundaries(con, clip::SourceClip{T}, col::String,
    sortrows::Int) where {T}
    from = "FROM read_parquet(?) $(windowfilter(col, clip.closed))"
    params = Any[clip.path, clip.start, clip.stop]
    counted = DBInterface.execute(con, "SELECT count(*) AS n $from", params)
    n = only(Tables.columntable(counted).n)
    n <= sortrows && return T[]
    k = min(cld(n, sortrows), MAXSLICES)
    qs = join((string(i / k) for i in 1:(k-1)), ", ")
    quantiles = try
        res =
            DBInterface.execute(con, "SELECT approx_quantile($col, [$qs]) AS q $from",
                params)
        collect(skipmissing(only(Tables.columntable(res).q)))
    catch e
        isduckdberror(e) || rethrow()
        Any[]
    end
    bounds = T[b for b in (slicebound(T, q) for q in quantiles) if b !== nothing]
    cuts = filter!(b -> clip.start < b < clip.stop, unique!(sort!(bounds)))
    slices = length(cuts) + 1
    2slices < k && @warn "readparquet: DuckDB found $(length(cuts)) $T cut times \
        inside the window in the time column of parquet file $(clip.path), so its \
        $n-row window is sorted in $(slices == 1 ? "one query" : "$slices queries") \
        rather than in $k slices of about `sortrows = $sortrows` rows"
    return cuts
end

# Each slice scans the file's time column, so a tiny `sortrows` would cost a
# scan per few rows: past this many slices they grow instead.
const MAXSLICES = 1024

# A quantile as a cut time in the context's time type, or nothing if it has
# none. A cut need only fall inside the window, so a fractional quantile (of a
# floating or decimal column) is floored into an integer context rather than
# refused, which would leave one slice.
function slicebound(::Type{T}, b) where {T}
    try
        return tobound(T, b)
    catch e
        e isa Union{MethodError,InexactError} || rethrow()
        return nothing
    end
end
tobound(::Type{T}, b) where {T} = convert(T, b)
tobound(::Type{T}, b::Real) where {T<:Integer} = floor(T, b)

# The SELECT list: the file columns a `columns` read keeps
# (`keptfilecolumns`), else `*`.
selectlist(filenames::Vector{String}, clip::SourceClip) =
    clip.columns === nothing ? "*" :
    join(map(quotename, keptfilecolumns(clip, filenames)), ", ")

quotename(name::String) = "\"" * replace(name, "\"" => "\"\"") * "\""

execstream(con, sql::String, params::Vector{Any}) =
    DBInterface.execute(DBInterface.prepare(con, sql, DuckDB.StreamResult), params)

# The file's column names, from metadata alone.
function columnnames(con, path::String)
    res = DBInterface.execute(con, "SELECT * FROM read_parquet(?) LIMIT 0",
        Any[path])
    return String[String(n) for n in Tables.schema(res).names]
end

# The write fallback: the stream is staged in a temporary table (which DuckDB
# spills to disk under memory pressure) and written by one COPY at the end.
function CausalFrames.parquetsink(::Val{:duckdb}, path::AbstractString,
    queue::Int, rowgroupsize::Int, opts::NamedTuple)
    # Translated on the pipeline's task, so an unsupported option is reported
    # when the run starts rather than inside the writer task.
    compression = copyoptions(opts)
    return ChunkSink(
        chan -> writeloop(chan, String(path), rowgroupsize, compression),
        queue, "writeparquet")
end

# The COPY options this backend can express. Other options are Parquet2's and
# are rejected rather than silently dropped.
function copyoptions(opts::NamedTuple)
    extra = filter(!=(:compression_codec), keys(opts))
    isempty(extra) || throw(ArgumentError("writeparquet: the DuckDB backend \
        does not support $(join(map(repr, extra), ", ")); those options belong \
        to the Parquet2 backend (run `using Parquet2`, or pass only \
        `compression_codec`)"))
    haskey(opts, :compression_codec) || return ""
    codec = uppercase(String(opts.compression_codec))
    return ", COMPRESSION $codec"
end

function writeloop(chan::Channel{DataFrame}, path::String, rowgroupsize::Int,
    compression::String)
    # Truncated when the run starts, as every sink's file is, so a failed or
    # abandoned run doesn't leave the previous run's file looking current.
    close(open(path, "w"))
    con = connection()
    staged = false
    try
        for c in chan
            DuckDB.register_table(con, c, "chunk")
            try
                DBInterface.execute(
                    con,
                    staged ?
                    "INSERT INTO staged SELECT * FROM chunk" :
                    "CREATE TEMP TABLE staged AS SELECT * FROM chunk",
                )
            finally
                DuckDB.unregister_table(con, "chunk")
            end
            staged = true
        end
        # A stream with no rows still leaves a valid, readable file.
        source = staged ? "staged" : "(SELECT NULL::BIGINT AS time WHERE FALSE)"
        DBInterface.execute(
            con,
            "COPY $source TO '$(quotepath(path))' \
(FORMAT parquet, ROW_GROUP_SIZE $rowgroupsize$compression)",
        )
    finally
        staged && DBInterface.execute(con, "DROP TABLE IF EXISTS staged")
    end
    return nothing
end

quotepath(path::String) = replace(path, "'" => "''")

end
