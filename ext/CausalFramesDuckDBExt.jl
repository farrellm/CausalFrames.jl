# The DuckDB backend: the only code in the package that names DuckDB. Reading
# (the preferred backend) is one streaming query per pipeline run, pulled one
# result chunk at a time, with the context window riding along as a WHERE clause
# so DuckDB skips the row groups and pages outside it. Writing (the fallback,
# used when Parquet2 is not loaded) stages the stream in a temporary table and
# writes the file with a single COPY, since DuckDB cannot append row groups to
# a parquet file.
module CausalFramesDuckDBExt

using CausalFrames
using DataFrames
using DuckDB
using Tables

using CausalFrames:
    ChunkSink, PullCursor, SourceClip, clipchunk!, gatherchunk!, pull!,
    sortgathered, timesourcename

CausalFrames.backendloaded(::Val{:duckdb}) = true

# One in-memory database for the process, created on first use (never at
# precompile time), plus a connection per pipeline run — a connection is
# single-consumer, and a pipeline may be run more than once (a self-join reads
# its file twice). Opening the database costs milliseconds, a connection
# microseconds.
const DBLOCK = ReentrantLock()
const DB = Ref{Any}(nothing)

function connection()
    return lock(DBLOCK) do
        DB[] === nothing && (DB[] = DuckDB.DB())
        return DBInterface.connect(DB[])
    end
end

# The stateful producer behind readparquet's ChunkSource, mirroring
# CSVProducer. The pull-to-pull state lives in fields rather than captured
# locals (captured variables that are reassigned get boxed). The dynamically
# typed fields are per-chunk setup state, not per-row state — the per-row
# `time` function runs behind a function barrier (`maptime`, via `clipchunk!`).
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
# whenever the time column can be named in SQL; everything else is clipped on
# arrival regardless, so a failure to push down costs only a longer read.
#
# A `sort` is pushed down the same way, as an ORDER BY on the time column. That
# sort is not stable in DuckDB, so ties are broken by `file_row_number`, the
# reader's virtual column (left out of `*`) — unless the file has a column of
# that name, which shadows it (names are case-insensitive in SQL). Without a
# nameable time column or a usable tiebreak, the rows are sorted in Julia.
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
        # The WHERE drops null times along with the rest of the outside, so
        # without `skipmissing` they go unreported here. An `OR $col IS NULL`
        # to surface them costs the row-group skip (see
        # notes/duckdb-null-pushdown.md).
        try
            execstream(p.con,
                "SELECT * FROM read_parquet(?) WHERE $col >= ? AND $col $upper ?$order",
                Any[clip.path, clip.start, clip.stop])
        catch
            # A time type DuckDB cannot bind, or cannot compare against this
            # column: read the whole file and let the clip do the work. A
            # genuinely unreadable file fails again below, with its own error.
            execstream(p.con, "SELECT * FROM read_parquet(?)$order", Any[clip.path])
        end
    end
    p.parts = PullCursor(Tables.partitions(res))
    return nothing
end

# The `sort = true` read the query could not sort: every result chunk is
# scanned for its in-window rows, which are sorted once, into a single chunk.
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

# The write fallback. DuckDB cannot append row groups to a parquet file, so the
# stream is staged in a temporary table — DuckDB spills it to disk under memory
# pressure — and written by one COPY when the stream ends.
function CausalFrames.parquetsink(::Val{:duckdb}, path::AbstractString,
    queue::Int, rowgroupsize::Int, opts::NamedTuple)
    # Translated here, on the pipeline's own task, so an unsupported option is
    # reported when the run starts rather than inside the writer task.
    compression = copyoptions(opts)
    return ChunkSink(
        chan -> writeloop(chan, String(path), rowgroupsize, compression),
        queue, "writeparquet")
end

# The COPY options this backend can express. Everything else is Parquet2's own,
# and silently dropping it would hide a request the user made deliberately.
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
    # Truncated when the run starts, as every other sink's file is: a run that
    # fails or is abandoned must not leave the previous run's file looking current.
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
