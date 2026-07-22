# The Parquet2 backend behind `writeparquet`: the only code in the package that
# names Parquet2. Row groups are written incrementally as the stream flows by,
# so the sink never holds more than `rowgroupsize` rows.
module CausalFramesParquet2Ext

using CausalFrames
using DataFrames
using Parquet2

using CausalFrames: ChunkSink

CausalFrames.backendloaded(::Val{:parquet2}) = true

function CausalFrames.parquetsink(path::AbstractString, queue::Int,
    rowgroupsize::Int, opts::NamedTuple)
    # Statistics on the time column are what readparquet's window pushdown
    # reads, so record them unless the caller says otherwise.
    o =
        haskey(opts, :compute_statistics) ? opts :
        merge(opts, (; compute_statistics = ["time"]))
    return ChunkSink(chan -> writeloop(chan, String(path), rowgroupsize, o),
        queue, "writeparquet")
end

# The background writer. One handle for the whole run; the footer is written by
# `finalize!` when the channel closes, which is the moment the file becomes
# readable at all.
function writeloop(chan::Channel{DataFrame}, path::String, rowgroupsize::Int,
    opts::NamedTuple)
    open(path, "w") do io
        fw = Parquet2.FileWriter(io, path; opts...)
        buffered = DataFrame[]
        rows = 0
        for c in chan
            push!(buffered, c)
            rows += nrow(c)
            if rows >= rowgroupsize
                emitgroup!(fw, buffered)
                rows = 0
            end
        end
        isempty(buffered) || emitgroup!(fw, buffered)
        Parquet2.finalize!(fw)
    end
    return nothing
end

# One row group from the pending chunks: they are only ever merged, never
# split, so a single pending chunk is written as it stands.
function emitgroup!(fw, buffered::Vector{DataFrame})
    Parquet2.writetable!(fw,
        length(buffered) == 1 ? only(buffered) : reduce(vcat, buffered))
    empty!(buffered)
    return nothing
end

end
