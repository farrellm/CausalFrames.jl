# Sources return a CausalPipeline; transforms are curried and return a
# CausalPipeline -> CausalPipeline function, so both chain with |>.
#
# Row functions (filterrows' pred, addcolumns' f) receive the concretely
# typed rows of a Tables.columntable, iterated behind a function barrier —
# never DataFrameRows, whose column accesses are type-unstable.

"""
    emptyframe() -> CausalPipeline

A source producing no rows. Loading it gives a zero-row frame with only
`:time`. It is the identity of [`concatenate`](@ref) and [`merge`](@ref "merge").
"""
emptyframe() = CausalPipeline(ctx -> ChunkSource(() -> nothing))

"""
    concatenate(ps::CausalPipeline...) -> CausalPipeline

A source running `ps` one after another over the same context and emitting
their rows end to end. Each pipeline starts only once the previous one is
exhausted, so a chain of file sources holds one file open at a time. With no
arguments it is [`emptyframe`](@ref). To interleave rows by time instead, use
[`merge`](@ref "merge").

# Arguments
- `ps`: the pipelines, in time order. Each is evaluated over the whole context
  and clips itself; keeping their data from overlapping is up to the caller.

Throws an `ArgumentError`, when the offending chunk arrives, if a pipeline's
column names differ from the first pipeline's, or if a pipeline's first time
precedes the previous one's last (equal times are allowed). Element types may
differ between pipelines; `DataFrame(frame)` promotes them.

```jldoctest
dir = mktempdir()
jan, feb = joinpath(dir, "jan.csv"), joinpath(dir, "feb.csv")
scan(Context(0, 10), readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5])) |> writecsv(jan))
scan(Context(0, 10), readtable(DataFrame(time = [3], bid = [11.0])) |> writecsv(feb))

types = Dict(:time => Int, :bid => Float64)
p = concatenate(readcsv(jan; types), readcsv(feb; types))
DataFrame(load(Context(0, 10), p))

# output

3×2 DataFrame
 Row │ time   bid
     │ Int64  Float64
─────┼────────────────
   1 │     1     10.0
   2 │     2     10.5
   3 │     3     11.0
```
"""
concatenate() = emptyframe()
concatenate(ps::CausalPipeline...) =
    CausalPipeline(ctx::Context -> ChunkSource(ConcatProducer(ps, ctx)))

# The stateful producer behind concatenate's ChunkSource. State lives in fields
# rather than captured locals, which would be boxed when reassigned. The untyped
# fields are touched once per chunk or per pipeline, never per row.
mutable struct ConcatProducer{P<:Tuple,C<:Context}
    const pipelines::P
    const ctx::C
    index::Int          # the pipeline currently being drained
    cursor::Any         # a PullCursor over its chunks; nothing until reached
    names::Union{Nothing,Vector{String}}  # column names of the first chunk
    prevtime::Any       # last time emitted, for cross-pipeline ordering
    previndex::Int      # which pipeline emitted it, for the error message
    ConcatProducer(ps::P, ctx::C) where {P<:Tuple,C<:Context} =
        new{P,C}(ps, ctx, 1, nothing, nothing, nothing, 0)
end

function (p::ConcatProducer)()
    while p.index <= length(p.pipelines)
        p.cursor === nothing &&
            (p.cursor = PullCursor(p.pipelines[p.index].run(p.ctx)))
        chunk = pull!(p.cursor)
        if chunk === nothing
            p.index += 1
            p.cursor = nothing
            continue
        end
        nrow(chunk) == 0 && continue
        checkconcat!(p, chunk::DataFrame)
        return chunk
    end
    return nothing
end

# O(ncols) per chunk. The chunk protocol already orders chunks within a
# pipeline, but checking every chunk is as cheap and lets the message name the
# pipelines at an out-of-order boundary.
function checkconcat!(p::ConcatProducer, c::DataFrame)
    cols = names(c)
    if p.names === nothing
        p.names = cols
    elseif p.names != cols
        throw(ArgumentError("concatenate: pipelines must have identical \
            columns; pipeline $(p.index) has $(cols), expected $(p.names)"))
    end
    t = first(c.time)
    p.prevtime === nothing || p.prevtime <= t ||
        throw(
            ArgumentError("concatenate: pipelines must be passed in time order; \
                pipeline $(p.index) starts at $t, before pipeline $(p.previndex) \
                ends at $(p.prevtime)"),
        )
    p.prevtime = last(c.time)
    p.previndex = p.index
    return nothing
end

"""
    clock(interval; batchsize = 1024) -> CausalPipeline

A source with only a `:time` column, one row at each of `start`,
`start + interval`, … before `stop`.

# Arguments
- `interval`: the tick spacing; anything that can be added to the time type
  (a `Dates.Period` for `DateTime`, a number for numeric time). Must be
  positive, checked when the pipeline runs.

# Keywords
- `batchsize = 1024`: rows per emitted chunk. Must be positive.
"""
function clock(interval; batchsize::Integer = 1024)
    batchsize > 0 ||
        throw(ArgumentError("clock batchsize must be positive, got $batchsize"))
    return CausalPipeline() do ctx::Context
        ctx.start + interval > ctx.start ||
            throw(ArgumentError("clock interval must be positive, got $interval"))
        return ClockChunks(ctx.start, ctx.stop, interval, Int(batchsize))
    end
end

struct ClockChunks{T,I}
    start::T
    stop::T
    interval::I
    batchsize::Int
end

Base.IteratorSize(::Type{<:ClockChunks}) = Base.SizeUnknown()
Base.eltype(::Type{<:ClockChunks}) = DataFrame

Base.iterate(it::ClockChunks) = iterate(it, it.start)
function Base.iterate(it::ClockChunks{T}, t) where {T}
    t < it.stop || return nothing
    times = T[]
    sizehint!(times, it.batchsize)
    while t < it.stop && length(times) < it.batchsize
        push!(times, t)
        t += it.interval
    end
    return (DataFrame(time = times), t)
end

"""
    readcsv(path; types = nothing, time = nothing, rename = nothing,
            delim = nothing, sort = false, chunkbytes = 4 * 1024 * 1024,
            closed = false, skipmissing = false) -> CausalPipeline

A source reading the CSV file at `path`, clipped to `[start, stop)`. The file is
read incrementally, in chunks, and reading stops at the first time past the
window. Column types are **not** inferred: every column is a `String` unless
`types` says otherwise.

# Arguments
- `path`: the file. A zero-byte file (what [`writecsv`](@ref) writes for an
  empty stream) reads as no rows.

# Keywords
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

Order and missing-time errors are raised only for the chunks actually read.
"""
function readcsv(path::AbstractString; types = nothing, time = nothing,
    rename = nothing, delim = nothing, sort::Bool = false,
    chunkbytes::Integer = 4 * 1024 * 1024, closed::Bool = false,
    skipmissing::Bool = false)
    checksourcetimespec(time, "readcsv")
    chunkbytes > 0 ||
        throw(ArgumentError("readcsv chunkbytes must be positive, got $chunkbytes"))
    # Fail eagerly when the time column is provably untyped: no `types`, or a
    # name-keyed dict with no entry for it. Other `types` forms, and any
    # `rename` (`types` names the file's original columns), can only be judged
    # on the first chunk, in `resolvetime!`.
    if !(time isa Function) && rename === nothing
        timename = time isa Symbol ? time : :time
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
            skipmissing)
        return ChunkSource(CSVProducer(clip, Int(chunkbytes), types, delim, sort))
    end
end

# The configuration and running state shared by every file source, and the
# argument `clipchunk!` and `gatherchunk!` take: where rows come from, how
# `:time` is resolved, which rows are kept, and, across chunks, the last raw
# time seen and whether the stream is done. A file-source keyword that bears on
# the clip is a field here. The untyped fields are read once per chunk; per-row
# work sits behind `clipchunk!`'s function barriers.
mutable struct SourceClip{T}
    const path::String
    const what::String  # the format, for messages: "CSV file", "parquet file", …
    const time::Any     # Nothing | Symbol (column name) | Function (row -> time)
    const rename::Any   # Nothing | AbstractDict/map | Function (name -> name)
    const closed::Bool
    const skipmissing::Bool
    const start::T
    const stop::T
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool          # a time past the window was seen, or the input ended
end

SourceClip(ctx::Context{T}, path::AbstractString, what::String, time, rename,
    closed::Bool, skipmissing::Bool) where {T} =
    SourceClip{T}(String(path), what, time, rename, closed, skipmissing,
        ctx.start, ctx.stop, nothing, false)

# The stateful producer behind readcsv's ChunkSource, with fields rather than
# boxed captured locals. The untyped fields are per-chunk setup; the per-row
# `time` function runs behind a function barrier (`maptime`).
mutable struct CSVProducer{T}
    const clip::SourceClip{T}
    const chunkbytes::Int
    const types::Any    # CSV.Chunks `types` argument, or nothing
    const delim::Any    # CSV.Chunks `delim` argument, or nothing
    const sort::Bool
    chunks::Any         # PullCursor over the file chunks, created on first pull
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

# The file's chunks. A file that fits in one chunk, or that CSV.Chunks refuses
# to split, is read whole. `delim = nothing` is CSV.jl's own default.
function csvchunks(path::String, chunkbytes::Int, types, delim)
    # writecsv leaves a zero-byte file (no header) for an empty stream.
    bytes = filesize(path)
    bytes == 0 && return DataFrame[]
    opts = (; types = typesfunction(types), stringtype = String, delim = delim)
    ntasks = max(1, Int(cld(bytes, chunkbytes)))
    ntasks == 1 && return [CSV.read(path, DataFrame; opts...)]
    try
        return CSV.Chunks(path; ntasks = ntasks, opts...)
    catch e
        e isa ArgumentError ? [CSV.read(path, DataFrame; opts...)] : rethrow()
    end
end

# Function barrier: computes the time column by applying `f` to the concretely
# typed rows of the column table, so `f` specializes and the eltype is inferred.
maptime(f, nt::NamedTuple) = map(f, Tables.rows(nt))

tochunk(filechunk) = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)

function (p::CSVProducer{T})() where {T}
    clip = p.clip
    clip.done && return nothing
    if p.sort
        clip.done = true
        kept = DataFrame[]
        for filechunk in csvchunks(clip.path, p.chunkbytes, p.types, p.delim)
            gatherchunk!(kept, clip, tochunk(filechunk))
        end
        return sortgathered(kept, T)
    end
    p.chunks === nothing &&
        (p.chunks = PullCursor(csvchunks(clip.path, p.chunkbytes, p.types, p.delim)))
    while !clip.done
        filechunk = pull!(p.chunks)
        filechunk === nothing && break
        out = clipchunk!(clip, tochunk(filechunk))
        out === nothing || return out
    end
    clip.done = true
    return nothing
end

# Shared by the file sources: rename columns, materialize `:time`, drop (or
# refuse) missing-time rows, check order within the chunk and against the
# previous one, clip to the window, and convert `:time` to the context's time
# type. Returns nothing when no row is in the window. A time past the window
# marks the clip done, which ends the source.
function clipchunk!(clip::SourceClip{T}, df::DataFrame) where {T}
    what, path = clip.what, clip.path
    renamecolumns!(df, clip.rename)
    resolvetime!(df, clip.time, path, what)
    # Missing rows are excluded through the row index rather than deleted
    # first, so the chunk is copied once.
    present = presentrows(df.time, clip.skipmissing, what, path)
    times = present === nothing ? df.time : view(df.time, present)
    issorted(times) || throw(ArgumentError(unordered(what, path)))
    if !isempty(times)
        clip.prevtime !== nothing && first(times) < clip.prevtime &&
            throw(ArgumentError(unordered(what, path)))
        clip.prevtime = last(times)
    end
    lo, hi = windowbounds(times, clip.closed, clip.start, clip.stop)
    hi < length(times) && (clip.done = true)   # saw a time past the window
    hi < lo && return nothing
    # The chunk is freshly read and owned, so keeping every row needs no copy.
    clipped =
        present !== nothing ? df[present[lo:hi], :] :
        lo == 1 && hi == nrow(df) ? df : df[lo:hi, :]
    clipped[!, :time] = convert(Vector{T}, clipped.time)
    return clipped
end

# `clipchunk!`'s counterpart for a source asked to `sort`: the same rename and
# time resolution, but with no order check, binary search or early stop. The
# in-window rows, found by a scan, are pushed onto `kept` to be sorted once the
# file is exhausted.
function gatherchunk!(kept::Vector{DataFrame}, clip::SourceClip, df::DataFrame)
    what, path = clip.what, clip.path
    renamecolumns!(df, clip.rename)
    resolvetime!(df, clip.time, path, what)
    present = presentrows(df.time, clip.skipmissing, what, path)
    closed, start, stop = clip.closed, clip.start, clip.stop
    rows =
        present === nothing ? windowrows(df.time, closed, start, stop) :
        present[windowrows(view(df.time, present), closed, start, stop)]
    if length(rows) == nrow(df)
        push!(kept, df)    # owned, as in clipchunk!
    elseif !isempty(rows)
        push!(kept, df[rows, :])
    end
    return nothing
end

# Function barrier, typed on the raw time vector: the indices of the rows in
# `[start, stop)`, or `[start, stop]` when closed, in file order.
windowrows(times::AbstractVector, closed::Bool, start, stop) =
    closed ? findall(t -> start <= t && t <= stop, times) :
    findall(t -> start <= t && t < stop, times)

# The gathered rows as one chunk stably sorted by time, so ties keep file
# order, with `:time` converted to the context's time type; nothing if empty.
function sortgathered(kept::Vector{DataFrame}, ::Type{T}) where {T}
    isempty(kept) && return nothing
    df = length(kept) == 1 ? only(kept) : reduce(vcat, kept)
    issorted(df.time) || (df = df[stableperm(df.time), :])
    df[!, :time] = convert(Vector{T}, df.time)
    return df
end

# The subject of every data error a source raises: "CSV file x.csv", or a bare
# "table" for readtable, which has no path. Built only on the error path.
sourcename(what::String, path::String) = isempty(path) ? what : "$what $path"

unordered(
    what::String,
    path::String,
) = "time column in $(sourcename(what, path)) is not non-decreasing"

# The error for a textual time column, which can't be ordered against the
# window. The hint depends on the cause: a `time` function to fix, CSV's
# `types`, or, for sources carrying their own types, a `time` function.
function textualtime(what::String, path::String, fromfunction::Bool)
    src = sourcename(what, path)
    fromfunction &&
        return "the `time` function over $src returned text; return a value \
            ordered like the context's times"
    what == "CSV file" &&
        return "time column in $src needs a concrete type via `types`, or a \
            `time` function to produce it"
    return "time column in $src is textual; use a `time` function to produce \
        a usable time"
end

# Whether the non-missing eltype is text. An all-missing column (nonmissingtype
# `Union{}`) is not.
function istextual(v::AbstractVector)
    S = nonmissingtype(eltype(v))
    return S !== Union{} && S <: AbstractString
end

# Rename columns before the time column is resolved. A map renames only the
# columns it names; a function is applied to every column name.
renamecolumns!(::DataFrame, ::Nothing) = nothing
renamecolumns!(df::DataFrame, f) = (rename!(f, df); nothing)
# A map may be keyed by String or Symbol names, so both are looked up.
function renamecolumns!(df::DataFrame, m::AbstractDict)
    pairs = Pair{String,Symbol}[]
    for n in names(df)
        s = Symbol(n)
        haskey(m, n) ? push!(pairs, n => Symbol(m[n])) :
        haskey(m, s) && push!(pairs, n => Symbol(m[s]))
    end
    isempty(pairs) || rename!(df, pairs)
    return nothing
end

# Materialize `:time`, from a column or a function, and reject it if textual.
# `columnindex`, unlike `names(df)`, allocates nothing.
function resolvetime!(df::DataFrame, time, path::String, what::String)
    if time isa Function
        df[!, :time] = maptime(time, Tables.columntable(df))
    else
        if time isa Symbol && time !== :time
            columnindex(df, time) > 0 ||
                throw(ArgumentError("$what $path has no column $(repr(time))"))
            columnindex(df, :time) > 0 &&
                throw(ArgumentError(timeclash(what, path, time)))
            rename!(df, time => :time)
        end
        columnindex(df, :time) > 0 || throw(
            ArgumentError("$what $path has no time column; choose one with `time`"))
    end
    istextual(df.time) &&
        throw(ArgumentError(textualtime(what, path, time isa Function)))
    return nothing
end

timeclash(
    what::String,
    path::String,
    name::Symbol,
) = "$(sourcename(what, path)) has both a :time column and the time column \
    $(repr(name)); drop one of them first"

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

checkqueue(queue::Integer, op::String) =
    queue >= 0 ||
    throw(ArgumentError("$op queue must be non-negative, got $queue"))

# Every file sink's transform: a pass-through chunkmap handing each chunk to a
# `ChunkSink` built by `makesink()` when the run starts (truncating the file
# then), and joining its writer once upstream is exhausted.
function sinktransform(makesink)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            sink = makesink()
            return chunkmap(c -> sinkchunk(sink, c), p.run(ctx);
                flush = () -> finishwrite(sink))
        end
    end
end

# Per-run writer state for every file sink: the queue feeding the background
# task, and the first chunk's column names, which fix the file's columns.
# `label` names the operator in errors.
mutable struct ChunkSink
    const chan::Channel{DataFrame}
    const task::Task
    const label::String
    names::Union{Nothing,Vector{String}}
end

function ChunkSink(writeloop, queue::Int, label::String)
    chan = Channel{DataFrame}(queue)
    task = Threads.@spawn writeloop(chan)
    # A failed writer closes the channel with its exception, so the pipeline
    # task sees it at the next put! rather than deadlocking on a full queue.
    bind(chan, task)
    return ChunkSink(chan, task, label, nothing)
end

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

function sinkchunk(sink::ChunkSink, c::DataFrame)
    cols = names(c)
    if sink.names === nothing
        sink.names = cols
    elseif sink.names != cols
        throw(ArgumentError("$(sink.label): chunk columns changed mid-stream, \
            from $(sink.names) to $(cols)"))
    end
    put!(sink.chan, c)
    # The writer reads `c` concurrently while downstream may mutate its chunk's
    # column index, so hand downstream a private index over the same vectors,
    # which are never mutated in place. O(ncols).
    return DataFrame(c; copycols = false)
end

# Called once, when upstream is exhausted: close the queue and join the writer,
# so the file is complete and closed by the time the stream ends.
function finishwrite(sink::ChunkSink)
    close(sink.chan)
    wait(sink.task)
    return nothing
end

"""
    filterrows(pred) -> (CausalPipeline -> CausalPipeline)
    filterrows(p::CausalPipeline, pred) -> CausalPipeline

A transform keeping the rows for which `pred(row)` is `true`.

# Arguments
- `pred`: a function `row -> Bool`. `row` supports `row.name` and `row[:name]`,
  including `row.time`.
"""
function filterrows(pred)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> filterchunk(pred, c), p.run(ctx))
        end
    end
end
filterrows(p::CausalPipeline, pred) = filterrows(pred)(p)

function filterchunk(pred, c::DataFrame)
    mask, kept = rowmask(pred, Tables.columntable(c))
    # The chunk is owned, so a filter that keeps every row needs no copy.
    return kept == length(mask) ? c : c[mask, :]
end

# Function barrier over concretely typed rows. Counting kept rows in the same
# pass saves a second walk for the "everything survived" test.
function rowmask(pred, nt::NamedTuple)
    mask = Vector{Bool}(undef, length(nt.time))
    kept = 0
    for (i, row) in enumerate(Tables.rows(nt))
        keep = pred(row)::Bool
        @inbounds mask[i] = keep
        kept += keep
    end
    return mask, kept
end

"""
    addcolumns(f) -> (CausalPipeline -> CausalPipeline)
    addcolumns(p::CausalPipeline, f) -> CausalPipeline

A transform appending columns computed from each row.

# Arguments
- `f`: a function `row -> NamedTuple`, where the `NamedTuple` maps each new
  column name to its value in that row. `row` is as for [`filterrows`](@ref).
  The names must be new, and may not include `time`; returning anything but a
  `NamedTuple` is an `ArgumentError`.

```jldoctest
p = readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7]))
DataFrame(load(Context(0, 10), p |> addcolumns(r -> (; mid = (r.bid + r.ask) / 2))))

# output

2×4 DataFrame
 Row │ time   bid      ask      mid
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     10.0     10.2     10.1
   2 │     2     10.5     10.7     10.6
```
"""
function addcolumns(f)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> addchunk(f, c), p.run(ctx))
        end
    end
end
addcolumns(p::CausalPipeline, f) = addcolumns(f)(p)

function addchunk(f, c::DataFrame)
    vals = rowvalues(f, Tables.columntable(c))
    first(vals) isa NamedTuple || throw(
        ArgumentError(
            "addcolumns function must return a NamedTuple, got $(typeof(first(vals)))"),
    )
    :time in keys(first(vals)) && throw(ArgumentError(
        "addcolumns function may not return a time column"))
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, DataFrame(vals); copycols = false)
end

# Function barrier: over concretely typed rows the comprehension infers a
# concrete NamedTuple eltype, so DataFrame builds typed columns directly.
rowvalues(f, nt::NamedTuple) = [f(row) for row in Tables.rows(nt)]

"""
    lag(offset) -> (CausalPipeline -> CausalPipeline)
    lag(p::CausalPipeline, offset) -> CausalPipeline

A transform moving every row `offset` later (`time -> time + offset`), so the
row at time `t` carries the values the input had at `t - offset`. Only `:time`
changes. The input runs over `[start - offset, stop - offset)`, so the output
fills the whole window.

# Arguments
- `offset`: the shift, in a type that can be added to and subtracted from the
  time type (a `Dates.Period`, a number). Must be non-negative, checked when the
  pipeline runs; `0` is the identity. For a forward shift, see
  [`Acausal.lead`](@ref CausalFrames.Acausal.lead).
"""
function lag(offset)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> shiftchunk!(c, offset), p.run(lagcontext(ctx, offset)))
        end
    end
end
lag(p::CausalPipeline, offset) = lag(offset)(p)

# The input window: the whole window slid back by `offset`, so the shift lands
# the output in [start, stop). A negative offset (acausal) is rejected by the
# same subtraction probe as `widenstart`.
function lagcontext(ctx::Context, offset)
    start = ctx.start - offset
    start <= ctx.start ||
        throw(ArgumentError("lag offset must be non-negative, got $offset"))
    return Context(start, ctx.stop - offset)
end

# Shift the owned chunk's time column by a constant, keeping its order and
# position. Shared with `Acausal.lead`, which passes a negative `delta`.
shiftchunk!(c::DataFrame, delta) = (c[!, :time] = shifttime(c.time, delta); c)

# Function barrier: the broadcast specializes on the concretely typed column.
shifttime(times::AbstractVector, delta) = times .+ delta

"""
    head(n) -> (CausalPipeline -> CausalPipeline)
    head(p::CausalPipeline, n) -> CausalPipeline

A transform emitting the first `n` rows of the window, then stopping: its input
is not pulled again, so `readcsv(path; types) |> head(10)` reads a single file
chunk.

# Arguments
- `n`: the number of rows, an `Integer`. Must be non-negative; `head(0)` never
  pulls its input.

`head` counts rows over the whole window, so `head(n)` over `[a, b)` and over
`[b, c)` can yield `2n` rows where `[a, c)` yields `n`.

Put writers downstream of `head`, never upstream (`p |> head(n) |>
writecsv(path)`): a writer finalizes its file only when its input is exhausted,
which `head` prevents.
"""
function head(n::Integer)
    n >= 0 || throw(ArgumentError("head n must be non-negative, got $n"))
    limit = Int(n)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return ChunkSource(HeadProducer(p.run(ctx), limit))
        end
    end
end
head(p::CausalPipeline, n::Integer) = head(n)(p)

# The stateful producer behind head's ChunkSource. head must stop pulling
# upstream once the budget is spent, which chunkmap (it drains its upstream)
# cannot do, so it drives the upstream cursor itself.
mutable struct HeadProducer{U}
    const upstream::PullCursor{U}
    remaining::Int
end
HeadProducer(upstream, n::Int) = HeadProducer(PullCursor(upstream), n)

function (p::HeadProducer)()
    p.remaining > 0 || return nothing
    chunk = pull!(p.upstream)
    if chunk === nothing
        p.remaining = 0     # ChunkSource requires nothing to be sticky
        return nothing
    end
    # The chunk protocol guarantees the annotation, which lets produce() infer
    # Union{Nothing, DataFrame} through the cursor's untyped state.
    return takerows!(p, chunk::DataFrame)
end

# A chunk that fits under the budget is owned and passed on uncopied; only the
# chunk that spends the budget is sliced.
function takerows!(p::HeadProducer, c::DataFrame)
    k = nrow(c)
    if k <= p.remaining
        p.remaining -= k
        return c
    end
    k = p.remaining
    p.remaining = 0
    return c[1:k, :]
end

"""
    settime(spec) -> (CausalPipeline -> CausalPipeline)
    settime(p::CausalPipeline, spec) -> CausalPipeline

A transform recomputing `:time`. The result is converted to the context's time
type and clipped to `[start, stop)`; other columns pass through.

# Arguments
- `spec`: either a `Symbol`, naming a column that replaces `:time` (taking its
  own position; the old `:time` is dropped), or a function `row -> time` whose
  result overwrites `:time` in place. `row` is as for [`filterrows`](@ref).

Each of these is an `ArgumentError` when the pipeline runs: a row whose new
time is earlier than its old one (see
[`Acausal.settime`](@ref CausalFrames.Acausal.settime)), and a new time column
that decreases within or across chunks.

The input is not widened: it runs over `[start, stop)`, so a row outside the
window is never seen, even if `spec` would move it inside. Hence loading
`[a, c)` need not equal loading `[a, b)` and `[b, c)`. `settime(:time)` changes
no values, but still drops rows at `stop`.
"""
function settime(spec)
    checktimespec(spec, "settime")
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            st = SetTimeState{timetype(ctx)}()
            return chunkmap(
                c -> settimechunk!(st, spec, c, ctx.start, ctx.stop, true, "settime"),
                p.run(ctx),
            )
        end
    end
end
settime(p::CausalPipeline, spec) = settime(spec)(p)

# Eager validation, shared by both settime variants and (through
# checksourcetimespec) the sources: a column name or a per-row function. It
# also turns a mistyped `settime(p)` into a clear message.
checktimespec(::Symbol, ::String) = nothing
checktimespec(::Function, ::String) = nothing
checktimespec(x, opname::String) = throw(
    ArgumentError(
        "invalid $opname spec of type $(typeof(x)): expected a column name \
        (Symbol) or a per-row function"),
)

# A source's `time`: the same spec, or `nothing` for the column named `:time`.
# Checked eagerly, since the chunk path would silently read `:time` for any
# other value.
checksourcetimespec(::Nothing, ::String) = nothing
checksourcetimespec(time, opname::String) = checktimespec(time, "$opname time")

# Per-run state. `prevtime` has the context's time type, so the cross-chunk
# comparison stays concrete. The inner constructor suppresses the default outer
# one, whose unbound T (called with `nothing`) fails Aqua's check.
mutable struct SetTimeState{T}
    prevtime::Union{Nothing,T}
    SetTimeState{T}() where {T} = new{T}(nothing)
end

# Shared by `settime` and `Acausal.settime`: recompute the owned chunk's :time
# from `spec`, validate it, and clip to [start, stop). `causal` adds the "no row
# moves earlier" check; `opname` names the operator in messages. The result may
# be empty, which chunkmap drops.
function settimechunk!(st::SetTimeState{T}, spec, c::DataFrame, start::T, stop::T,
    causal::Bool, opname::String) where {T}
    old = c.time
    new = converttimes(T, newtimes(spec, c, opname), opname)
    causal && checkforward(old, new, opname)
    issorted(new) ||
        throw(ArgumentError("$opname produced a time column that is not non-decreasing"))
    st.prevtime === nothing || st.prevtime <= first(new) ||
        throw(
            ArgumentError("$opname produced a time column that is not non-decreasing \
                across chunk boundaries"),
        )
    st.prevtime = last(new)          # pre-clip, as clipchunk! carries it
    lo = searchsortedfirst(new, start)
    hi = searchsortedfirst(new, stop) - 1
    settimecolumn!(c, spec, new)
    # The chunk is owned, so keeping every row needs no copy.
    return lo == 1 && hi == nrow(c) ? c : c[lo:hi, :]
end

# The raw new time values. A textual result is rejected, as in `resolvetime!`.
function newtimes(spec::Symbol, c::DataFrame, opname::String)
    String(spec) in names(c) ||
        throw(ArgumentError("$opname: no column named $(repr(spec))"))
    return checktimevalues(c[!, spec], opname)
end
newtimes(spec::Function, c::DataFrame, opname::String) =
    checktimevalues(maptime(spec, Tables.columntable(c)), opname)

checktimevalues(v::AbstractVector, opname::String) =
    eltype(v) <: AbstractString ?
    throw(
        ArgumentError("$opname produced a textual time column (element type \
            $(eltype(v))); parse it to an ordered type first"),
    ) : v

# The new column takes the context's time type, as every source's does.
function converttimes(::Type{T}, times::AbstractVector, opname::String) where {T}
    eltype(times) <: T && return times
    return convert(Vector{T}, times)
end

# Function barrier over concretely typed vectors. An explicit loop, rather than
# `all(new .>= old)`, allocates nothing and knows the offending row.
# `eachindex(old, new)` asserts equal lengths.
function checkforward(old::AbstractVector, new::AbstractVector, opname::String)
    @inbounds for i in eachindex(old, new)
        new[i] >= old[i] || throw(
            ArgumentError(
                "$opname may not move a row earlier in time: row $i moves from \
                $(old[i]) to $(new[i]); use `CausalFrames.Acausal.settime` for that"),
        )
    end
    return nothing
end

# The Symbol form drops the old :time and renames the named column to :time,
# in its own position. The Function form overwrites :time in place. Both mutate
# the owned chunk's column index (see DESIGN.md, "CSV output").
function settimecolumn!(c::DataFrame, spec::Symbol, new::AbstractVector)
    if spec !== :time
        select!(c, Not(:time))
        rename!(c, spec => :time)
    end
    c[!, :time] = new
    return c
end
settimecolumn!(c::DataFrame, ::Function, new::AbstractVector) = (c[!, :time] = new; c)

"""
    selectcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    selectcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform keeping only the selected columns, in their input order. `:time` is
always kept.

# Arguments
- `selectors`: at least one column selector: a name (`Symbol` or `String`),
  a `Regex` matched against names, a predicate called with the name as a
  `String`, or a collection of these. A column is kept if any selector matches
  it. A name the data lacks is an `ArgumentError`; a `Regex` or predicate
  matching nothing is not.

```jldoctest
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> selectcolumns(:venue, r"^px_", startswith("qty"))
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "px_bid"
 "px_ask"
 "qty_bid"
 "venue"
```
"""
function selectcolumns(selectors...)
    checkselectors(selectors, "selectcolumns", false)
    return columntransform(cols -> keptcolumns(selectors, cols, true,
        "selectcolumns"))
end
selectcolumns(p::CausalPipeline, selectors...) = selectcolumns(selectors...)(p)

"""
    dropcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    dropcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform removing the selected columns and keeping the rest, in their input
order.

# Arguments
- `selectors`: at least one column selector, as for
  [`selectcolumns`](@ref). A column is dropped if any selector matches it.
  `:time` is never dropped: a `Regex` or predicate matching it is ignored, and
  naming it is an `ArgumentError`, as is naming a column the data lacks.
"""
function dropcolumns(selectors...)
    checkselectors(selectors, "dropcolumns", true)
    return columntransform(cols -> keptcolumns(selectors, cols, false,
        "dropcolumns"))
end
dropcolumns(p::CausalPipeline, selectors...) = dropcolumns(selectors...)(p)

"""
    reordercolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    reordercolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform moving the selected columns to the front, after `:time`, in the
order of the selectors. The other columns follow in their input order.

# Arguments
- `selectors`: at least one column selector, as for
  [`selectcolumns`](@ref). Nested collections are flattened in place. A `Regex`
  or predicate contributes its matches in input order, and a column matched
  twice goes where it was first matched. `:time` always stays first: a `Regex`
  or predicate matching it is ignored, and naming it is an `ArgumentError`, as
  is naming a column the data lacks.

```jldoctest
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> reordercolumns(:venue, r"^qty_")   # time, venue, qty_bid, then the rest
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "venue"
 "qty_bid"
 "px_bid"
 "px_ask"
```
"""
function reordercolumns(selectors...)
    checkselectors(selectors, "reordercolumns", false)
    foreachliteral(selectors) do n
        n == "time" && throw(
            ArgumentError("reordercolumns: the time column is always first"))
    end
    return columntransform(cols -> orderedcolumns(selectors, cols))
end
reordercolumns(p::CausalPipeline, selectors...) = reordercolumns(selectors...)(p)

# The column transforms' shared shape: a chunkmap indexing each owned chunk by
# `resolve(names)`, the columns to keep in order, or `nothing` to pass it
# through.
function columntransform(resolve)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            memo = SchemaMemo()
            return chunkmap(p.run(ctx)) do c
                keep = memoized!(resolve, memo, c)
                return keep === nothing ? c : c[!, keep]
            end
        end
    end
end

# The resolution, cached against the names it came from. The schema rarely
# changes, so this skips re-running the selectors, while re-resolving when names
# differ keeps validation per-chunk (load and stream don't recheck schemas).
mutable struct SchemaMemo
    lastnames::Union{Nothing,Vector{String}}
    resolved::Union{Nothing,Vector{Symbol}}
end
SchemaMemo() = SchemaMemo(nothing, nothing)

function memoized!(resolve, memo::SchemaMemo, c::DataFrame)
    cols = names(c)
    memo.lastnames == cols && return memo.resolved
    memo.resolved = resolve(cols)
    memo.lastnames = cols
    return memo.resolved
end

# The names to keep, in the chunk's own column order, or `nothing` when every
# column survives.
function keptcolumns(selectors::Tuple, cols::Vector{String}, selecting::Bool,
    opname::String)
    foreachliteral(selectors) do n
        n in cols || throw(
            ArgumentError("$opname: no column named $(repr(Symbol(n)))"))
    end
    keep = Symbol[]
    for n in cols
        (n == "time" || matchescolumn(selectors, n) == selecting) &&
            push!(keep, Symbol(n))
    end
    return length(keep) == length(cols) ? nothing : keep
end

# The column order to impose, or `nothing` when the chunk already has it.
# `placed` tracks columns by position, avoiding hashing.
function orderedcolumns(selectors::Tuple, cols::Vector{String})
    foreachliteral(selectors) do n
        n in cols || throw(
            ArgumentError("reordercolumns: no column named $(repr(Symbol(n)))"))
    end
    order = Symbol[:time]
    placed = falses(length(cols))
    for (i, n) in pairs(cols)
        n == "time" && (placed[i] = true)
    end
    foreachselector(selectors) do s
        for (i, n) in pairs(cols)
            placed[i] && continue
            matchescolumn(s, n) || continue
            push!(order, Symbol(n))
            placed[i] = true
        end
    end
    for (i, n) in pairs(cols)
        placed[i] || push!(order, Symbol(n))
    end
    return all(i -> order[i] === Symbol(cols[i]), eachindex(cols)) ? nothing :
           order
end

# Numbers and Chars iterate as themselves, so the collection fallbacks below
# would recurse forever on them.
const ScalarSelector = Union{Number,Char}

selectorerror(x) = throw(
    ArgumentError("invalid column selector of type $(typeof(x)): expected a \
        name, a Regex, a predicate, or a collection of those"))

checkselector(selectors) =
    applicable(iterate, selectors) || selectorerror(selectors)

# Whether a selector spec matches a column name. Leaves dispatch first, so a
# predicate is never taken for a collection; anything else must be iterable
# and is matched recursively.
matchescolumn(s::Symbol, name::AbstractString) = String(s) == name
matchescolumn(s::AbstractString, name::AbstractString) = String(s) == name
matchescolumn(r::Regex, name::AbstractString) = occursin(r, name)
matchescolumn(f::Function, name::AbstractString) = f(name)::Bool
matchescolumn(x::ScalarSelector, ::AbstractString) = selectorerror(x)
function matchescolumn(selectors, name::AbstractString)
    checkselector(selectors)
    return any(s -> matchescolumn(s, name), selectors)
end

# Walk the name leaves of a selector spec, ignoring regex and predicate ones.
foreachliteral(f, s::Symbol) = (f(String(s)); nothing)
foreachliteral(f, s::AbstractString) = (f(String(s)); nothing)
foreachliteral(::Any, ::Regex) = nothing
foreachliteral(::Any, ::Function) = nothing
foreachliteral(::Any, x::ScalarSelector) = selectorerror(x)
function foreachliteral(f, selectors)
    checkselector(selectors)
    for s in selectors
        foreachliteral(f, s)
    end
    return nothing
end

# Walk every leaf of a selector spec in order, including the regex and
# predicate leaves `foreachliteral` skips.
foreachselector(f, s::Symbol) = (f(s); nothing)
foreachselector(f, s::AbstractString) = (f(s); nothing)
foreachselector(f, r::Regex) = (f(r); nothing)
foreachselector(f, g::Function) = (f(g); nothing)
foreachselector(::Any, x::ScalarSelector) = selectorerror(x)
function foreachselector(f, selectors)
    checkselector(selectors)
    for s in selectors
        foreachselector(f, s)
    end
    return nothing
end

# Eager validation: at least one selector, every leaf usable, and, for
# dropcolumns, no literal `:time`.
function checkselectors(selectors::Tuple, opname::String, dropping::Bool)
    isempty(selectors) &&
        throw(ArgumentError("$opname requires at least one column selector"))
    foreachliteral(selectors) do n
        dropping && n == "time" &&
            throw(ArgumentError("$opname may not drop the time column"))
    end
    return nothing
end
