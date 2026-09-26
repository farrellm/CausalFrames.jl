# The Parquet2 backend: the only code in the package that names Parquet2.
# Writing (the preferred backend) sends row groups off incrementally as the
# stream flows by, so the sink never holds more than `rowgroupsize` rows.
# Reading (the fallback, used when DuckDB is not loaded) walks the file's own
# row groups, skipping those whose recorded time statistics put them outside
# the context window.
module CausalFramesParquet2Ext

using CausalFrames
using DataFrames
using Parquet2

using CausalFrames:
    ChunkSink, SourceClip, clipchunk!, gatherchunk!, sortgathered,
    timesourcename

CausalFrames.backendloaded(::Val{:parquet2}) = true

function CausalFrames.parquetsink(::Val{:parquet2}, path::AbstractString,
    queue::Int, rowgroupsize::Int, opts::NamedTuple)
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
        next = iterate(chan)
        if next === nothing
            # A stream with no rows gets the DuckDB sink's empty `time` column,
            # since DuckDB cannot read a parquet file with no columns at all.
            # Statistics are off: Parquet2 cannot compute them over zero rows.
            fw = Parquet2.FileWriter(io, path; opts..., compute_statistics = false)
            Parquet2.writetable!(fw, DataFrame(time = Int64[]))
            Parquet2.finalize!(fw)
            return nothing
        end
        fw = Parquet2.FileWriter(io, path; opts...)
        buffered = DataFrame[]
        rows = 0
        for c in Iterators.flatten(((first(next),), Iterators.rest(chan, next[2])))
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

# The read fallback. One row group per chunk, in file order; the pull-to-pull
# state lives in fields rather than captured locals (captured variables that are
# reassigned get boxed), and the dynamically typed fields are per-chunk setup
# state, not per-row state.
mutable struct RowGroupProducer{T}
    const clip::SourceClip{T}
    const sort::Bool
    dataset::Any        # Parquet2.Dataset, opened on first pull
    timecol::Any        # file-level name of the time column, or nothing
    index::Int          # next row group
    usestats::Bool      # statistics comparable with this context's times
end

CausalFrames.parquetproducer(::Val{:parquet2}, clip::SourceClip{T},
    sort::Bool) where {T} =
    RowGroupProducer{T}(clip, sort, nothing, nothing, 1, true)

function (p::RowGroupProducer)()
    clip = p.clip
    clip.done && return nothing
    p.dataset === nothing && open!(p)
    p.sort && return sortedread!(p)
    while !clip.done && p.index <= Parquet2.nrowgroups(p.dataset)
        rg = p.index
        p.index += 1
        skip = rowgroupwindow(p, rg)
        skip === :after && break
        skip === :before && continue
        out = clipchunk!(clip, DataFrame(p.dataset[rg]))
        out === nothing || return out
    end
    clip.done = true
    return nothing
end

# The `sort = true` read: a file out of time order has no row group after which
# nothing more can be in the window, so every group is visited — statistics
# still skip the ones wholly outside it, on either side — and the in-window rows
# are sorted once, into the stream's single chunk.
function sortedread!(p::RowGroupProducer{T}) where {T}
    p.clip.done = true
    kept = DataFrame[]
    for rg in 1:Parquet2.nrowgroups(p.dataset)
        rowgroupwindow(p, rg) === :overlaps || continue
        gatherchunk!(kept, p.clip, DataFrame(p.dataset[rg]))
    end
    return sortgathered(kept, T)
end

function open!(p::RowGroupProducer)
    p.dataset = Parquet2.Dataset(p.clip.path)
    p.timecol = timesourcename(String[String(n) for n in Base.names(p.dataset)],
        p.clip.time, p.clip.rename)
    return nothing
end

# Where a row group sits relative to the window, from its recorded statistics
# alone: `:before` (skippable), `:after` (so are all later ones, the file being
# non-decreasing), or `:overlaps` — which is also the answer whenever the
# statistics are missing or unusable, since skipping is only ever an
# optimization. A closed window keeps a group starting exactly at `stop`.
function rowgroupwindow(p::RowGroupProducer, rg::Int)
    (p.usestats && p.timecol !== nothing) || return :overlaps
    stats = Parquet2.ColumnStatistics(Parquet2.Column(p.dataset, rg, p.timecol))
    lo, hi = minimum(stats), maximum(stats)
    (lo === nothing || hi === nothing) && return :overlaps
    try
        clip = p.clip
        hi < clip.start && return :before
        (clip.closed ? lo > clip.stop : lo >= clip.stop) && return :after
    catch
        # Times this context cannot compare against: stop consulting statistics
        # for the rest of the run rather than failing over an optimization.
        p.usestats = false
    end
    return :overlaps
end

end
