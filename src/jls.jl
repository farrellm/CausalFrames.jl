# The JLS sink and source: a stream persisted through the Serialization stdlib,
# as a header record followed by one serialized DataFrame per chunk. Unlike CSV
# and parquet, which are typed formats, this round-trips whatever a column
# holds — at the price of a file tied to the Julia and package versions that
# wrote it. The sink reuses the file sinks' ChunkSink machinery whole; the
# source is a CSVProducer-shaped producer over the shared clipchunk!.

# The first record of every file. A NamedTuple of isbits values serializes
# identically across Julia versions, so a foreign or future file is reported
# as such rather than as an opaque deserialization failure.
const JLSHEADER = (format = :CausalFramesJLS, version = 1)

"""
    writejls(path; queue = 1) -> (CausalPipeline -> CausalPipeline)
    writejls(p::CausalPipeline, path; queue = 1) -> CausalPipeline

A transparent pass-through transform that writes the stream to `path` through
Julia's `Serialization` stdlib as it flows by, yielding every chunk downstream
unchanged — [`writecsv`](@ref)'s contract in a format that stores any Julia
value a column holds, so columns that neither CSV nor parquet can encode (a
fitted model, a `NamedTuple`) round-trip through [`readjls`](@ref).

Each chunk becomes one serialized record, written and flushed as it is
produced on a background task fed by a bounded queue of depth `queue`, exactly
as for `writecsv`. The file is truncated when the run starts and finalized when
the stream is exhausted; a stream with no rows yields a file holding only its
header, which reads back as an empty stream. Every complete record of an
interrupted run stays readable.

A JLS file is readable only by a compatible Julia and compatible versions of the
packages whose types it holds, so it is a persistence format for a pipeline's
own outputs, not an interchange format.

The curried form composes with `|>`; the uncurried form applies directly, so
`writejls(p, path)` is equivalent to `p |> writejls(path)`.
"""
function writejls(path::AbstractString; queue::Integer = 1)
    queue >= 0 ||
        throw(ArgumentError("writejls queue must be non-negative, got $queue"))
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            sink = ChunkSink(chan -> jlswriteloop(chan, String(path)),
                Int(queue), "writejls")
            return chunkmap(c -> sinkchunk(sink, c), p.run(ctx);
                flush = () -> finishwrite(sink))
        end
    end
end
writejls(p::CausalPipeline, path::AbstractString; kwargs...) =
    writejls(path; kwargs...)(p)

# The background writer. Each record is its own `serialize` call, so records
# carry no back-references to each other and the reader can take them one at a
# time; flushing after each keeps a complete prefix on disk.
function jlswriteloop(chan::Channel{DataFrame}, path::String)
    open(path, "w") do io
        serialize(io, JLSHEADER)
        flush(io)
        for c in chan
            serialize(io, c)
            flush(io)
        end
    end
    return nothing
end

"""
    readjls(path) -> CausalPipeline

A source reading a file written by [`writejls`](@ref): one chunk per record,
each clipped to the context's `[start, stop)`, with the time converted to the
context's time type. Reading is incremental — a record at a time, never the
whole file — and stops as soon as a time `>= stop` is seen, but there is no
index to seek by, so a read costs the file's prefix up to `stop`. As for
[`readcsv`](@ref), sortedness is checked in the records actually read.

A file that was not written by `writejls` is an `ArgumentError`, as is one whose
last record was cut short by an interrupted write (every record before it has
already been emitted). Deserialization can construct arbitrary types, so read
only files you trust.
"""
function readjls(path::AbstractString)
    return CausalPipeline() do ctx::Context
        return ChunkSource(JLSProducer{timetype(ctx)}(String(path), ctx.start,
            ctx.stop))
    end
end

# The stateful producer behind readjls's ChunkSource, in CSVProducer's shape:
# the pull-to-pull state lives in fields rather than reassigned closure
# captures (which would be boxed). `prevtime` is dynamically typed, but it is
# per-chunk setup state; the per-row work is clipchunk!'s.
mutable struct JLSProducer{T}
    const path::String
    const start::T
    const stop::T
    io::Union{Nothing,IOStream}   # opened on the first pull
    prevtime::Any                 # last raw time seen, for cross-chunk order
    done::Bool
    JLSProducer{T}(path, start, stop) where {T} =
        new{T}(path, start, stop, nothing, nothing, false)
end

function (p::JLSProducer{T})() where {T}
    p.done && return nothing
    p.io === nothing && (p.io = openjls(p.path))
    io = p.io::IOStream
    while true
        if eof(io)
            finishjls!(p)
            return nothing
        end
        df = readrecord(io, p.path)
        df isa DataFrame || (
            finishjls!(p);
            throw(
                ArgumentError(
                    "jls file $(p.path) holds a $(typeof(df)) record where a chunk was expected",
                ),
            )
        )
        clipped, sawstop, p.prevtime = clipchunk!(df, nothing, nothing, p.path,
            "jls file", p.prevtime, p.start, p.stop)
        sawstop && finishjls!(p)
        nrow(clipped) > 0 && return clipped
        p.done && return nothing
    end
end

function finishjls!(p::JLSProducer)
    p.done = true
    p.io === nothing || close(p.io)
    return nothing
end

# Open the file and consume its header, turning every way that can fail into an
# ArgumentError naming the file.
function openjls(path::String)
    io = open(path, "r")
    header = try
        eof(io) ? nothing : deserialize(io)
    catch
        nothing
    end
    if !(header isa NamedTuple && get(header, :format, nothing) === JLSHEADER.format)
        close(io)
        throw(ArgumentError("$path is not a jls file written by writejls"))
    end
    if header.version != JLSHEADER.version
        close(io)
        throw(ArgumentError("jls file $path has format version \
            $(header.version); this CausalFrames reads version $(JLSHEADER.version)"))
    end
    return io
end

# One record. A torn last record (an interrupted writejls) surfaces as an
# EOFError from inside Serialization; report it as what it is.
function readrecord(io::IOStream, path::String)
    try
        return deserialize(io)
    catch e
        e isa EOFError || rethrow()
        close(io)
        throw(ArgumentError("jls file $path ends in a truncated record \
            (an interrupted writejls?)"))
    end
end
