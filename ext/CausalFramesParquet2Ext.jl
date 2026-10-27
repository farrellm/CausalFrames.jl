# The Parquet2 backend, the only code that names Parquet2. Writing (preferred)
# writes row groups as the stream flows by, buffering about `rowgroupsize`
# rows. Reading (the fallback) walks the file's row groups, skipping those
# whose time statistics put them outside the window.
module CausalFramesParquet2Ext

using CausalFrames
using DataFrames
using Parquet2

using CausalFrames:
    ChunkSink, PullCursor, SourceClip, pullclipped!, keptfilecolumns, timesourcename,
    vcatowned

CausalFrames.backendloaded(::Val{:parquet2}) = true

function CausalFrames.parquetsink(::Val{:parquet2}, path::AbstractString,
    queue::Int, rowgroupsize::Int, opts::NamedTuple)
    # readparquet's pushdown reads the time column's statistics, so record
    # them unless the caller says otherwise.
    o =
        haskey(opts, :compute_statistics) ? opts :
        merge(opts, (; compute_statistics = ["time"]))
    return ChunkSink(chan -> writeloop(chan, String(path), rowgroupsize, o),
        queue, "writeparquet")
end

# The background writer: one handle for the run. `finalize!` writes the footer
# when the channel closes; only then is the file readable.
function writeloop(chan::Channel{DataFrame}, path::String, rowgroupsize::Int,
    opts::NamedTuple)
    open(path, "w") do io
        next = iterate(chan)
        if next === nothing
            # An empty stream gets an empty `time` column, as from the DuckDB
            # sink, since DuckDB can't read a parquet file with no columns.
            # Parquet2 can't compute statistics over zero rows.
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

# One row group from the pending chunks, which are merged, never split.
function emitgroup!(fw, buffered::Vector{DataFrame})
    Parquet2.writetable!(fw, vcatowned(buffered))
    empty!(buffered)
    return nothing
end

# The read fallback: one row group per chunk, in file order. The untyped fields
# are per-chunk setup.
mutable struct RowGroupProducer{T}
    const clip::SourceClip{T}
    const sort::Bool
    dataset::Any        # Parquet2.Dataset, opened on first pull
    timecol::Any        # file-level name of the time column, or nothing
    groups::Any         # PullCursor over the picked row groups, as DataFrames
    usestats::Bool      # statistics comparable with this context's times
end

# `sortrows` is DuckDB's: a sorted read here gathers the whole window.
CausalFrames.parquetproducer(::Val{:parquet2}, clip::SourceClip{T},
    sort::Bool, ::Int) where {T} =
    RowGroupProducer{T}(clip, sort, nothing, nothing, nothing, true)

function (p::RowGroupProducer)()
    p.groups === nothing && open!(p)
    return pullclipped!(p.clip, p.groups, p.sort)
end

# Open the file and pick its row groups by their statistics. In a time-ordered
# file the groups before the window are skipped and the first one after it ends
# the read. A `sort = true` read can't rely on the order, so it reads every
# group overlapping the window.
function open!(p::RowGroupProducer)
    p.dataset = Parquet2.Dataset(p.clip.path)
    filenames = String[String(n) for n in Base.names(p.dataset)]
    p.timecol = timesourcename(filenames, p.clip.time, p.clip.rename)
    kept = keptfilecolumns(p.clip, filenames)
    placed = Iterators.map(rg -> (rg, rowgroupwindow(p, rg)),
        1:Parquet2.nrowgroups(p.dataset))
    picked =
        p.sort ? Iterators.filter(g -> g[2] === :overlaps, placed) :
        Iterators.filter(g -> g[2] !== :before,
            Iterators.takewhile(g -> g[2] !== :after, placed))
    p.groups = PullCursor(
        Iterators.map(g -> rowgroupframe(p.dataset[g[1]], p.clip, kept), picked))
    return nothing
end

# A row group's columns, all of them or, for a `columns` read, only those it
# keeps, loaded as Parquet2 loads a whole row group (in parallel per
# `Parquet2.useparallel`).
rowgroupframe(rg, clip::SourceClip, kept::Vector{String}) =
    clip.columns === nothing ? DataFrame(rg) :
    DataFrame(Parquet2.TableOperations.select(rg, map(Symbol, kept)...);
        copycols = false)

# Where a row group sits relative to the window, by its statistics: `:before`,
# `:after` (so are all later ones, in a time-ordered file) or `:overlaps`, the
# answer too when statistics are missing or unusable. A closed window keeps a
# group starting exactly at `stop`.
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
        # Statistics incomparable with this context's times: stop consulting
        # them for the rest of the run.
        p.usestats = false
    end
    return :overlaps
end

end
