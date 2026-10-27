"""
    readcsv(path; columns = nothing, types = nothing, time = nothing,
            rename = nothing, delim = nothing, sort = false,
            chunkbytes = 4 * 1024 * 1024, closed = false, skipmissing = false)
        -> CausalPipeline

A source reading the CSV file at `path`, clipped to `[start, stop)`. The file is
read incrementally, in chunks, and reading stops at the first time past the
window. Column types are **not** inferred: every column is a `String` unless
`types` says otherwise.

# Arguments
- `path`: the file. A zero-byte file (what [`writecsv`](@ref) writes for an
  empty stream) reads as no rows.

# Keywords
- `columns = nothing`: the columns to read, as a vector or tuple of `Symbol`s
  named after `rename`; the others are never parsed. `:time` (or the column
  `time` names) is always read and need not be listed, and a `time` function
  sees only the columns read. The columns keep the file's order. Duplicates, or
  anything but `Symbol`s, are an `ArgumentError`, as is a listed column the
  file lacks.
- `types = nothing`: concrete types for some columns, keyed by the file's own
  column names (before `rename`): a single type for every column, a vector
  indexed by column position (`nothing` entries stay `String`), a `Dict` keyed by
  name (`Symbol` or `String`) or position, or a function `(index, name) -> type`
  returning `nothing` for `String`. Unless `time` is a function, the time column
  must be given a type here.
- `time = nothing`: where `:time` comes from — `nothing` for the column named
  `:time`; a `Symbol` naming another column (after `rename`), which is renamed
  to `:time` and must not coexist with one; or a function `row -> time`,
  whose result replaces any existing `:time`. The result is converted to the
  context's time type.
- `rename = nothing`: a map (`Dict` from old to new name) or a function
  `name -> name`, applied to the column names after typing and before `time` is
  resolved.
- `delim = nothing`: the field delimiter (`Char` or `String`), passed to CSV.jl;
  `nothing` auto-detects.
- `sort = false`: stably sort the rows by time, for a file not stored in time
  order. Sorting reads the whole file and emits its in-window rows as one chunk,
  so memory scales with the window. Without it, a decreasing time is an
  `ArgumentError`.
- `chunkbytes = 4 * 1024 * 1024`: the approximate size of each chunk read.
  Must be positive.
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`.
- `skipmissing = false`: drop rows whose time is `missing` (a blank cell, or a
  `time` function returning `missing`). Without it such a row is an
  `ArgumentError`.

Unlike [`readparquet`](@ref), there is no `backend` (CSV.jl reads every file)
and no `sortrows` (a sort gathers the window). Order and missing-time errors are raised only for the chunks actually
read.
"""
function readcsv(path::AbstractString; columns = nothing, types = nothing,
    time = nothing, rename = nothing, delim = nothing, sort::Bool = false,
    chunkbytes::Integer = 4 * 1024 * 1024, closed::Bool = false,
    skipmissing::Bool = false)
    checksourcetimespec(time, "readcsv")
    names = sourcecolumns(columns, "readcsv")
    checkpositive(chunkbytes, "readcsv chunkbytes")
    # Fail eagerly when the time column is provably untyped: no `types`, or a
    # name-keyed dict with no entry for it. Other `types` forms, and any
    # `rename` (`types` names the file's original columns), can only be judged
    # on the first chunk, in `resolvetime!`.
    if !(time isa Function) && rename === nothing
        timename = timetarget(time)
        namekeyed = types isa AbstractDict && keytype(types) <: Union{Symbol,
            AbstractString}
        typedbydict =
            namekeyed &&
            (haskey(types, timename) || haskey(types, String(timename)))
        (types === nothing || (namekeyed && !typedbydict)) &&
            throw(ArgumentError("readcsv time column $(repr(timename)) needs a \
                concrete type via `types`, or a `time` function to produce it"))
    end
    return CausalPipeline() do ctx::Context
        clip = SourceClip(ctx, path, "CSV file", time, rename, closed,
            skipmissing, names)
        return ChunkSource(CSVProducer(clip, Int(chunkbytes), types, delim, sort))
    end
end

# The stateful producer behind readcsv's ChunkSource, with fields rather than
# boxed captured locals. The untyped fields are per-chunk setup; the per-row
# `time` function runs behind a function barrier (`maptime`).
mutable struct CSVProducer{T}
    const clip::SourceClip{T}
    const chunkbytes::Int
    const types::Any    # CSV.Chunks `types` argument, or nothing
    const delim::Any    # CSV.Chunks `delim` argument, or nothing
    const sort::Bool
    chunks::Any         # PullCursor over the file's DataFrames, created on first pull
end
CSVProducer(clip::SourceClip{T}, chunkbytes::Int, types, delim,
    sort::Bool) where {T} =
    CSVProducer{T}(clip, chunkbytes, types, delim, sort, nothing)

# The user's `types` as a CSV.jl `types` function `(index, name::Symbol)` that
# defaults every unspecified column to `String`, so nothing is inferred.
typesfunction(::Nothing) = (i, name) -> String
typesfunction(t::Type) = (i, name) -> t
typesfunction(v::AbstractVector) =
    (i, name) -> (1 <= i <= length(v) && v[i] !== nothing) ? v[i] : String
typesfunction(f) = (i, name) -> something(f(i, name), String)  # user function
function typesfunction(d::AbstractDict)
    return function (i, name)
        haskey(d, name) && return d[name]
        haskey(d, String(name)) && return d[String(name)]
        haskey(d, i) && return d[i]
        return String
    end
end

# CSV.jl's `select` for a `columns` read: the header's names that every reader
# reads (`keptfilecolumns`), from a read of the header alone.
function csvselect(clip::SourceClip, delim)
    header = CSV.File(clip.path; limit = 0, delim = delim, types = String)
    filenames = String[String(n) for n in propertynames(header)]
    return map(Symbol, keptfilecolumns(clip, filenames))
end

# The file's chunks. A file that fits in one chunk, or that CSV.Chunks refuses
# to split, is read whole. `delim = nothing` is CSV.jl's own default.
function csvchunks(clip::SourceClip, chunkbytes::Int, types, delim)
    path = clip.path
    # writecsv leaves a zero-byte file (no header) for an empty stream.
    bytes = filesize(path)
    bytes == 0 && return DataFrame[]
    opts = (; types = typesfunction(types), stringtype = String, delim = delim,
        select = clip.columns === nothing ? nothing : csvselect(clip, delim))
    ntasks = max(1, Int(cld(bytes, chunkbytes)))
    ntasks == 1 && return [CSV.read(path, DataFrame; opts...)]
    try
        return CSV.Chunks(path; ntasks = ntasks, opts...)
    catch e
        e isa ArgumentError ? [CSV.read(path, DataFrame; opts...)] : rethrow()
    end
end

tochunk(filechunk) = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)

function (p::CSVProducer)()
    clip = p.clip
    p.chunks === nothing && (
        p.chunks = PullCursor(
            Iterators.map(tochunk,
                csvchunks(clip, p.chunkbytes, p.types, p.delim)),
        )
    )
    return pullclipped!(clip, p.chunks, p.sort)
end

"""
    writecsv(path; queue = 1, kwargs...) -> (CausalPipeline -> CausalPipeline)
    writecsv(p::CausalPipeline, path; queue = 1, kwargs...) -> CausalPipeline

A pass-through transform writing every chunk to the CSV file at `path` as it
flows by, then yielding it downstream unchanged. Each chunk is written and
flushed by a background task, so the file grows while the pipeline runs.

# Arguments
- `path`: the file, truncated when the pipeline starts running. An empty stream
  leaves it empty, which [`readcsv`](@ref) reads back as no rows.

# Keywords
- `queue = 1`: how many chunks may wait for the writer before the pipeline
  blocks; `0` hands each chunk over directly. Must be non-negative.
- `kwargs...`: passed to `CSV.write` (`delim`, `missingstring`, `dateformat`,
  …). `append`, `header`, `writeheader`, `partition` and `compress` are
  controlled by `writecsv` and are an `ArgumentError`.

The file is complete only once the stream is exhausted — by [`load`](@ref),
[`scan`](@ref) or a fully drained [`stream`](@ref). A chunk whose column names
differ from the first chunk's is an `ArgumentError`. Use `scan` when the file
is all you want:

```jldoctest
path = joinpath(mktempdir(), "mids.csv")
readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7])) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
    writecsv(path) |>
    scan(Context(0, 10))
print(read(path, String))

# output

time,bid,ask,mid
1,10.0,10.2,10.1
2,10.5,10.7,10.6
```
"""
function writecsv(path::AbstractString; queue::Integer = 1, kwargs...)
    checkqueue(queue, "writecsv")
    for k in (:append, :header, :writeheader, :partition, :compress)
        haskey(kwargs, k) && throw(ArgumentError("writecsv controls the \
            $(repr(k)) option of CSV.write itself; it may not be passed"))
    end
    # A concretely typed NamedTuple, for the per-chunk splat into CSV.write.
    opts = values(kwargs)
    return sinktransform(
        () -> ChunkSink(
            chan -> csvwriteloop(chan, String(path), opts), Int(queue), "writecsv"),
    )
end
writecsv(p::CausalPipeline, path::AbstractString; kwargs...) =
    writecsv(path; kwargs...)(p)

# The background CSV writer: one handle for the run, closed when the channel
# closes. Each chunk is flushed as it lands, so an interrupted run leaves a
# complete prefix. `append = false` on the first chunk writes the header once.
function csvwriteloop(chan::Channel{DataFrame}, path::String, opts::NamedTuple)
    open(path, "w") do io
        first = true
        for c in chan
            CSV.write(io, c; append = !first, opts...)
            flush(io)
            first = false
        end
    end
    return nothing
end
