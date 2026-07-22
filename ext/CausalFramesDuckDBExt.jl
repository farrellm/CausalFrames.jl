# The DuckDB backend behind `readparquet`: the only code in the package that
# names DuckDB. A read is one streaming query per pipeline run, pulled one
# result chunk at a time; the context window rides along as a WHERE clause so
# DuckDB skips the row groups and pages outside it.
module CausalFramesDuckDBExt

using CausalFrames
using DataFrames
using DuckDB
using Tables

using CausalFrames: Context, clipchunk!, timesourcename, timetype

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
    const path::String
    const start::T
    const stop::T
    const time::Any     # Nothing | Symbol (column name) | Function (row -> time)
    const rename::Any   # Nothing | AbstractDict/map | Function (name -> name)
    con::Any            # DuckDB connection, opened on first pull
    parts::Any          # result chunk iterator
    state::Any          # its iteration state
    started::Bool
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool
    ParquetProducer{T}(path, start, stop, time, rename) where {T} =
        new{T}(path, start, stop, time, rename, nothing, nothing, nothing,
            false, nothing, false)
end

CausalFrames.parquetproducer(ctx::Context, path::AbstractString, time, rename) =
    ParquetProducer{timetype(ctx)}(String(path), ctx.start, ctx.stop, time, rename)

function (p::ParquetProducer{T})() where {T}
    p.done && return nothing
    p.parts === nothing && startquery!(p)
    while true
        next = p.started ? iterate(p.parts, p.state) : iterate(p.parts)
        if next === nothing
            p.done = true
            return nothing
        end
        chunk, p.state = next
        p.started = true
        clipped, sawstop, p.prevtime = clipchunk!(DataFrame(chunk), p.time,
            p.rename, p.path, "parquet file", p.prevtime, p.start, p.stop)
        sawstop && (p.done = true)
        nrow(clipped) > 0 && return clipped
        p.done && return nothing
    end
end

# Open the connection and start the streaming scan. The window is pushed down
# whenever the time column can be named in SQL; everything else is clipped on
# arrival regardless, so a failure to push down costs only a longer read.
function startquery!(p::ParquetProducer)
    p.con = connection()
    src = timesourcename(columnnames(p.con, p.path), p.time, p.rename)
    res = if src === nothing
        execstream(p.con, "SELECT * FROM read_parquet(?)", Any[p.path])
    else
        col = replace(src, "\"" => "\"\"")
        try
            execstream(p.con,
                "SELECT * FROM read_parquet(?) WHERE \"$col\" >= ? AND \"$col\" < ?",
                Any[p.path, p.start, p.stop])
        catch
            # A time type DuckDB cannot bind, or cannot compare against this
            # column: read the whole file and let the clip do the work. A
            # genuinely unreadable file fails again below, with its own error.
            execstream(p.con, "SELECT * FROM read_parquet(?)", Any[p.path])
        end
    end
    p.parts = Tables.partitions(res)
    return nothing
end

execstream(con, sql::String, params::Vector{Any}) =
    DBInterface.execute(DBInterface.prepare(con, sql, DuckDB.StreamResult), params)

# The file's column names, from metadata alone.
function columnnames(con, path::String)
    res = DBInterface.execute(con, "SELECT * FROM read_parquet(?) LIMIT 0",
        Any[path])
    return String[String(n) for n in Tables.schema(res).names]
end

end
