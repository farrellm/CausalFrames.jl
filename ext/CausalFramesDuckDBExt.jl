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
    ChunkSink, PullCursor, SourceClip, clipchunk!, gatherchunk!, pull!,
    sortgathered, timesourcename

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
mutable struct ParquetProducer{T}
    const clip::SourceClip{T}
    const sort::Bool
    gather::Bool        # sort in Julia, the query being unable to
    con::Any            # DuckDB connection, opened on first pull
    parts::Any          # PullCursor over the result chunks
end

CausalFrames.parquetproducer(::Val{:duckdb}, clip::SourceClip{T},
    sort::Bool) where {T} =
    ParquetProducer{T}(clip, sort, false, nothing, nothing)

function (p::ParquetProducer)()
    clip = p.clip
    clip.done && return nothing
    p.parts === nothing && startquery!(p)
    p.gather && return gatherread!(p)
    while !clip.done
        chunk = pull!(p.parts)
        chunk === nothing && break
        out = clipchunk!(clip, DataFrame(chunk))
        out === nothing || return out
    end
    clip.done = true
    return nothing
end

# Open the connection and start the streaming scan. The window is pushed down
# whenever the time column can be named in SQL; chunks are clipped on arrival
# regardless, so a failed pushdown only costs a longer read.
#
# A `sort` is pushed down too, as an ORDER BY on the time column. DuckDB's sort
# is unstable, so ties are broken by `file_row_number`, the reader's virtual
# column (not in `*`), unless a file column of that name (case-insensitively)
# shadows it. Without a nameable time column or a usable tiebreak, the rows are
# sorted in Julia.
function startquery!(p::ParquetProducer)
    clip = p.clip
    p.con = connection()
    filenames = columnnames(p.con, clip.path)
    src = timesourcename(filenames, clip.time, clip.rename)
    p.gather =
        p.sort &&
        (src === nothing || any(n -> lowercase(n) == "file_row_number", filenames))
    res = if src === nothing
        execstream(p.con, "SELECT * FROM read_parquet(?)", Any[clip.path])
    else
        col = "\"" * replace(src, "\"" => "\"\"") * "\""
        order = p.sort && !p.gather ? " ORDER BY $col, file_row_number" : ""
        upper = clip.closed ? "<=" : "<"
        # The WHERE also drops null times, so without `skipmissing` they go
        # unreported here; `OR $col IS NULL` would cost the row-group skip
        # (see notes/duckdb-null-pushdown.md).
        try
            execstream(p.con,
                "SELECT * FROM read_parquet(?) WHERE $col >= ? AND $col $upper ?$order",
                Any[clip.path, clip.start, clip.stop])
        catch
            # A time type DuckDB can't bind or compare with this column: read
            # the whole file and let the clip do the work. An unreadable file
            # fails again here with its own error.
            execstream(p.con, "SELECT * FROM read_parquet(?)$order", Any[clip.path])
        end
    end
    p.parts = PullCursor(Tables.partitions(res))
    return nothing
end

# A `sort = true` read the query couldn't sort: every result chunk is scanned
# for in-window rows, which are sorted once into a single chunk.
function gatherread!(p::ParquetProducer{T}) where {T}
    p.clip.done = true
    kept = DataFrame[]
    for chunk in p.parts.iter
        gatherchunk!(kept, p.clip, DataFrame(chunk))
    end
    return sortgathered(kept, T)
end

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
