# The JLS sink and source: a stream persisted with the Serialization stdlib as
# a header record followed by one serialized DataFrame per chunk. Unlike CSV
# and parquet it round-trips any column value, but the file is tied to the
# Julia and package versions that wrote it. The sink uses ChunkSink; the source
# clips with the shared clipchunk!.

# The first record of every file. A NamedTuple of a Symbol and an Int
# serializes stably across Julia versions, so a foreign or future file is
# reported as such rather than failing opaquely.
const JLSHEADER = (format = :CausalFramesJLS, version = 1)

"""
    writejls(path; queue = 1) -> (CausalPipeline -> CausalPipeline)
    writejls(p::CausalPipeline, path; queue = 1) -> CausalPipeline

A pass-through transform writing every chunk to `path` with Julia's
`Serialization`, then yielding it downstream unchanged. It stores any value a
column can hold — fitted models, `NamedTuple`s — which CSV and parquet cannot.
Read the file back with [`readjls`](@ref).

# Arguments
- `path`: the file, truncated when the pipeline starts running.

# Keywords
- `queue = 1`: how many chunks may wait for the writer before the pipeline
  blocks; `0` hands each chunk over directly. Must be non-negative.

Each chunk is one record, written and flushed on a background task. The file is
complete once the stream is exhausted, but every record written before an
interruption stays readable. A JLS file can be read only with a compatible
Julia and compatible versions of the packages whose types it holds.
"""
function writejls(path::AbstractString; queue::Integer = 1)
    checkqueue(queue, "writejls")
    return sinktransform(
        () -> ChunkSink(
            chan -> jlswriteloop(chan, String(path)), Int(queue), "writejls"),
    )
end
writejls(p::CausalPipeline, path::AbstractString; kwargs...) =
    writejls(path; kwargs...)(p)

# The background writer. Each record is its own `serialize` call, so records
# share no back-references and can be read one at a time; flushing each keeps a
# complete prefix on disk.
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
    readjls(path; closed = false) -> CausalPipeline

A source reading a file written by [`writejls`](@ref), one chunk per record,
clipped to `[start, stop)`. Records are read one at a time, stopping at the
first time past the window; there is no index, so a read costs the file up to
`stop`.

# Arguments
- `path`: the file. One not written by `writejls`, or whose last record is
  truncated, is an `ArgumentError`. Deserialization can construct any type, so
  read only files you trust.

# Keywords
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`.
"""
function readjls(path::AbstractString; closed::Bool = false)
    return CausalPipeline() do ctx::Context
        # a written stream's times are resolved and never missing
        clip = SourceClip(ctx, path, "jls file", nothing, nothing, closed, false)
        return ChunkSource(JLSProducer(clip))
    end
end

# The stateful producer behind readjls's ChunkSource.
mutable struct JLSProducer{T}
    const clip::SourceClip{T}
    io::Union{Nothing,IOStream}   # opened on the first pull
end
JLSProducer(clip::SourceClip{T}) where {T} = JLSProducer{T}(clip, nothing)

function (p::JLSProducer)()
    clip = p.clip
    clip.done && return nothing
    p.io === nothing && (p.io = openjls(clip.path))
    io = p.io::IOStream
    while !clip.done && !eof(io)
        df = readrecord(io, clip.path)
        df isa DataFrame || (
            finishjls!(p);
            throw(
                ArgumentError(
                    "jls file $(clip.path) holds a $(typeof(df)) record where a chunk was expected",
                ),
            )
        )
        out = clipchunk!(clip, df)
        clip.done && finishjls!(p)
        out === nothing || return out
    end
    finishjls!(p)
    return nothing
end

function finishjls!(p::JLSProducer)
    p.clip.done = true
    p.io === nothing || close(p.io)
    return nothing
end

# Open the file and consume its header; any failure is an ArgumentError naming
# the file.
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

# One record. A truncated last record (an interrupted writejls) surfaces as an
# EOFError, reported as such.
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
