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

# The stateful producer behind concatenate's ChunkSource, in the shape of
# readcsv's CSVProducer: the pull-to-pull state lives in fields rather than
# captured locals (captured variables that are reassigned get boxed). The
# dynamically typed fields are per-pipeline setup state — one dynamic index
# into the (heterogeneous) pipeline tuple and one iterator hand-off per
# pipeline; nothing here runs per row.
mutable struct ConcatProducer{P<:Tuple,C<:Context}
    const pipelines::P
    const ctx::C
    index::Int          # the pipeline currently being drained
    chunks::Any         # its chunk iterator; nothing until it is reached
    state::Any          # its iteration state
    started::Bool
    names::Union{Nothing,Vector{String}}  # column names of the first chunk
    prevtime::Any       # last time emitted, for cross-pipeline ordering
    previndex::Int      # which pipeline emitted it, for the error message
    ConcatProducer(ps::P, ctx::C) where {P<:Tuple,C<:Context} =
        new{P,C}(ps, ctx, 1, nothing, nothing, false, nothing, nothing, 0)
end

function (p::ConcatProducer)()
    while p.index <= length(p.pipelines)
        if p.chunks === nothing
            p.chunks = p.pipelines[p.index].run(p.ctx)
            p.started = false
        end
        next = p.started ? iterate(p.chunks, p.state) : iterate(p.chunks)
        if next === nothing
            p.index += 1
            p.chunks = nothing
            continue
        end
        chunk, p.state = next
        p.started = true
        nrow(chunk) == 0 && continue
        checkconcat!(p, chunk)
        return chunk
    end
    return nothing
end

# O(ncols) per chunk. Within one pipeline the chunk protocol already
# guarantees the ordering; checking every chunk costs nothing extra and lets
# the message name the pipelines when a boundary is the one out of order.
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
    # Eager error where the time column is provably untyped: no `types` at all,
    # or a name-keyed `types` dict that has no entry for the time column. Other
    # `types` forms (positional vectors, index-keyed dicts, functions) can only
    # be judged once the columns are realized, so they defer to the first-chunk
    # check in `resolvetime!`. `types` names the *original* CSV columns, so a
    # `rename` breaks the name correspondence too — defer that case as well.
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
        return ChunkSource(
            CSVProducer{timetype(ctx)}(String(path), Int(chunkbytes),
                ctx.start, ctx.stop, types, time, rename, delim, sort, closed,
                skipmissing),
        )
    end
end

# The stateful producer behind readcsv's ChunkSource. The pull-to-pull state
# lives in fields rather than captured locals (captured variables that are
# reassigned get boxed). The dynamically typed fields are per-chunk setup
# state, not per-row state — the per-row `time` function runs behind a
# function barrier (`maptime`).
mutable struct CSVProducer{T}
    const path::String
    const chunkbytes::Int
    const start::T
    const stop::T
    const types::Any    # CSV.Chunks `types` argument, or nothing
    const time::Any     # Nothing | Symbol (column name) | Function (row -> time)
    const rename::Any   # Nothing | AbstractDict/map | Function (name -> name)
    const delim::Any    # CSV.Chunks `delim` argument, or nothing
    const sort::Bool
    const closed::Bool
    const skipmissing::Bool
    chunks::Any         # file-chunk iterator, created on first pull
    state::Any          # its iteration state
    started::Bool
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool
    CSVProducer{T}(path, chunkbytes, start, stop, types, time, rename,
        delim, sort, closed, skipmissing) where {T} =
        new{T}(path, chunkbytes, start, stop, types, time, rename, delim, sort,
            closed, skipmissing, nothing, nothing, false, nothing, false)
end

# The user's `types` (or nothing) as a CSV.jl per-column `types` function that
# defaults every unspecified column to `String` — so nothing is ever inferred.
# CSV calls it with a 1-based column index and a `Symbol` name.
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

# CSV.Chunks refuses files it cannot split (ntasks == 1, or too few rows to
# justify it); such a file fits in one chunk, so read it whole. Columns are
# read as plain `String` (`stringtype`) unless `types` overrides them.
function csvchunks(path::String, chunkbytes::Int, types, delim)
    # CSV.jl's own default for `delim` is `nothing`, so passing it through
    # unchanged is a no-op.
    # A zero-byte file is what writecsv leaves for a stream with no rows (with
    # no chunk, there is no header to write), so it reads back as one.
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

function (p::CSVProducer{T})() where {T}
    p.done && return nothing
    p.chunks === nothing &&
        (p.chunks = csvchunks(p.path, p.chunkbytes, p.types, p.delim))
    if p.sort
        p.done = true
        kept = DataFrame[]
        for filechunk in p.chunks
            df = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)
            gatherchunk!(kept, df, p.time, p.rename, p.path, "CSV file",
                p.closed, p.skipmissing, p.start, p.stop)
        end
        return sortgathered(kept, T)
    end
    while true
        next = p.started ? iterate(p.chunks, p.state) : iterate(p.chunks)
        if next === nothing
            p.done = true
            return nothing
        end
        filechunk, p.state = next
        p.started = true
        df = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)
        clipped, sawstop, p.prevtime = clipchunk!(df, p.time, p.rename, p.path,
            "CSV file", p.prevtime, p.closed, p.skipmissing, p.start, p.stop)
        sawstop && (p.done = true)
        nrow(clipped) > 0 && return clipped
        p.done && return nothing
    end
end

# Shared by the file sources: rename the columns, materialize `:time`, drop (or
# refuse) the rows whose time is missing, check sortedness within the chunk and
# against the last time of the previous one, clip to [start, stop) (or
# [start, stop] when closed), and convert `:time` to the context's time type.
# `what` names the format in error messages. Returns the clipped chunk (which
# may have no rows), whether a time past the window was seen (the source is then
# done), and the last raw time of this chunk, to be carried to the next call.
function clipchunk!(df::DataFrame, time, rename, path::String, what::String,
    prevtime, closed::Bool, skipmissing::Bool, start::T, stop::T) where {T}
    renamecolumns!(df, rename)
    resolvetime!(df, time, path, what)
    # Missing rows are folded into the clip's row index rather than deleted
    # first, so a chunk holding some costs one copy of its in-window rows.
    present = presentrows(df.time, skipmissing, what, path)
    times = present === nothing ? df.time : view(df.time, present)
    issorted(times) || throw(ArgumentError(unordered(what, path)))
    if !isempty(times)
        prevtime !== nothing && first(times) < prevtime &&
            throw(ArgumentError(unordered(what, path)))
        prevtime = last(times)
    end
    lo, hi = windowbounds(times, closed, start, stop)
    sawstop = hi < length(times)   # saw a time past the window
    # The chunk is freshly materialized and owned, so a clip that keeps every
    # row needs no copy.
    clipped =
        present !== nothing ? df[present[lo:hi], :] :
        lo == 1 && hi == nrow(df) ? df : df[lo:hi, :]
    clipped[!, :time] = convert(Vector{T}, clipped.time)
    return (clipped, sawstop, prevtime)
end

# `clipchunk!`'s counterpart for a source asked to `sort`, shared by readcsv and
# readparquet: the same rename and time resolution, but the chunk's order is
# not the file's promise, so there is no order check, no binary search and no
# early stop — the in-window rows are found by a scan and pushed onto `kept`
# (when there are any), to be sorted once the file is exhausted.
function gatherchunk!(kept::Vector{DataFrame}, df::DataFrame, time, rename,
    path::String, what::String, closed::Bool, skipmissing::Bool, start, stop)
    renamecolumns!(df, rename)
    resolvetime!(df, time, path, what)
    present = presentrows(df.time, skipmissing, what, path)
    rows =
        present === nothing ? windowrows(df.time, closed, start, stop) :
        present[windowrows(view(df.time, present), closed, start, stop)]
    if length(rows) == nrow(df)
        push!(kept, df)    # freshly materialized and owned, as in clipchunk!
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

# The gathered rows as one stably time-sorted chunk (nothing when none were in
# the window), `:time` converted to the context's time type. Chunks are gathered
# in file order, so stability across them is file order too.
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

# A time column that arrived as text cannot be ordered against the window. A
# `time` function that produced it has to be fixed itself; otherwise only CSV
# has a `types` knob to point the user at, and parquet or a table carries its
# own types, so there the way out is a `time` function.
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

# Text times, blank cells aside (a CSV text column with blanks is
# `Union{Missing,String}`); an all-missing column is not text, whose
# `nonmissingtype` is `Union{}`.
function istextual(v::AbstractVector)
    S = nonmissingtype(eltype(v))
    return S !== Union{} && S <: AbstractString
end

# Rename columns before the time column is resolved. A map renames only the
# columns it names; a function is applied to every column name.
renamecolumns!(::DataFrame, ::Nothing) = nothing
renamecolumns!(df::DataFrame, f) = (rename!(f, df); nothing)
# A map may be keyed by either the column name as a String or as a Symbol, so
# both are looked up; normalizing the pairs to one type up front keeps the two
# cases from having to be collected separately and spliced together.
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

# Materialize the `:time` column and check it is usable (not text), whether it
# was a column or produced by a function. `what` names the file format for error
# messages. Columns are looked up by `columnindex`, which unlike `names(df)`
# allocates nothing.
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
    queue >= 0 ||
        throw(ArgumentError("writecsv queue must be non-negative, got $queue"))
    for k in (:append, :header, :writeheader, :partition, :compress)
        haskey(kwargs, k) && throw(ArgumentError("writecsv controls the \
            $(repr(k)) option of CSV.write itself; it may not be passed"))
    end
    # Materialized once, so the per-chunk splat into CSV.write is over a
    # concretely typed NamedTuple rather than the keyword iterator.
    opts = values(kwargs)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            sink = ChunkSink(chan -> csvwriteloop(chan, String(path), opts),
                Int(queue), "writecsv")
            return chunkmap(c -> sinkchunk(sink, c), p.run(ctx);
                flush = () -> finishwrite(sink))
        end
    end
end
writecsv(p::CausalPipeline, path::AbstractString; kwargs...) =
    writecsv(path; kwargs...)(p)

# Per-run writer state, shared by every file sink: the queue feeding the
# background task, plus the column names of the first chunk, which pin the
# file's columns. Per-run mutable state lives here rather than in reassigned
# closure captures (which get boxed). `label` names the operator in errors.
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

# The background CSV writer. One handle for the whole run, closed
# deterministically when the channel closes; each chunk is flushed as it lands,
# so an interrupted run still leaves a complete prefix on disk. `append` is
# false only for the first chunk, which is what makes CSV.write emit the header
# exactly once.
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
    # The writer reads `c` concurrently, while downstream transforms may mutate
    # their chunk's column index in place (they own what they are handed), so
    # give them a private index over the same column vectors — those are never
    # mutated in place, only replaced wholesale. O(ncols), nothing per row.
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

# Function barrier: iterates concretely typed rows of the column table. The
# kept count falls out of the same pass, so the caller's "everything survived"
# test costs no second walk over the mask.
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

# Function barrier: with concretely typed rows the comprehension infers, so
# the collected values have a concrete NamedTuple eltype and DataFrame builds
# typed columns from them directly.
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

# A negative offset would shift rows earlier, making lag acausal; reject it at
# run time the way rightcontext validates tolerance (probe start - offset <=
# start). This is the mirror of asofjoin's rightcontext: the whole window slides
# back by the offset so the +offset shift lands the output in [start, stop).
function lagcontext(ctx::Context, offset)
    start = ctx.start - offset
    start <= ctx.start ||
        throw(ArgumentError("lag offset must be non-negative, got $offset"))
    return Context(start, ctx.stop - offset)
end

# Shift the owned chunk's time column by a constant, preserving order (so no
# re-sort) and column position. Shared with the acausal `lead`. `delta` may be
# negative (lead subtracts). One allocation per chunk for the new column.
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

# The stateful producer behind head's ChunkSource. head must genuinely stop
# pulling upstream once the budget is spent, which a chunkmap cannot do — its
# `advance` loops until upstream returns nothing — so it drives the upstream
# iterator itself, in the shape of readcsv's CSVProducer: the pull-to-pull state
# lives in fields rather than captured locals (captured variables that are
# reassigned get boxed). The dynamically typed `state` field is touched once per
# chunk, never per row, exactly as ConcatProducer's is.
mutable struct HeadProducer{U}
    const upstream::U
    remaining::Int
    state::Any          # the upstream iteration state
    started::Bool
end
HeadProducer(upstream::U, n::Int) where {U} = HeadProducer{U}(upstream, n, nothing, false)

function (p::HeadProducer)()
    p.remaining > 0 || return nothing
    next = p.started ? iterate(p.upstream, p.state) : iterate(p.upstream)
    p.started = true
    if next === nothing
        p.remaining = 0     # ChunkSource requires nothing to be sticky
        return nothing
    end
    chunk, p.state = next
    # The chunk protocol guarantees the annotation, which is what lets produce()
    # infer Union{Nothing, DataFrame} through the dynamically typed state.
    return takerows!(p, chunk::DataFrame)
end

# The chunk is owned and its column vectors are never mutated in place, so one
# that fits entirely under the budget is passed on as it is — filterchunk's and
# clipchunk!'s "keeps everything, so no copy" rule. Only the chunk that spends
# the budget is sliced.
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

# Eager validation, shared by the two settime variants and (through
# checksourcetimespec) the sources: a column name or a per-row function, nothing
# else. A CausalPipeline lands here too, which is what turns a
# mistyped `settime(p)` into a message rather than a MethodError deep in a chunk.
checktimespec(::Symbol, ::String) = nothing
checktimespec(::Function, ::String) = nothing
checktimespec(x, opname::String) = throw(
    ArgumentError(
        "invalid $opname spec of type $(typeof(x)): expected a column name \
        (Symbol) or a per-row function"),
)

# A source's `time`: the same spec, or `nothing` for the column named `:time`.
# Checked eagerly, since the chunk path only acts on a Symbol or a Function and
# would read the `:time` column in place of anything else it was given.
checksourcetimespec(::Nothing, ::String) = nothing
checksourcetimespec(time, opname::String) = checktimespec(time, "$opname time")

# Per-run mutable state, in a field rather than a reassigned closure capture
# (those get boxed). Unlike CSVProducer's `prevtime::Any` the type is known
# here — it is the context's — so the cross-chunk comparison stays concrete.
# The constructor is inner, as ConcatProducer's is, which suppresses the default
# outer one: `SetTimeState(prevtime::Union{Nothing,T}) where {T}` cannot bind T
# when called with `nothing`, and Aqua's unbound-type-parameter check fails on it.
mutable struct SetTimeState{T}
    prevtime::Union{Nothing,T}
    SetTimeState{T}() where {T} = new{T}(nothing)
end

# Shared by the causal `settime` and `Acausal.settime`, the way `shiftchunk!` is
# shared with `lead`: recompute the owned chunk's :time from `spec`, validate the
# result, and clip to [start, stop). `causal` adds the per-row "no row moves
# earlier" rule; `opname` names the operator in the messages, which is the only
# other thing the two variants disagree about. Returns the clipped chunk, which
# may have no rows — chunkmap drops those.
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
    # The chunk is owned, so a clip that keeps every row needs no copy — the
    # rule filterchunk and clipchunk! follow.
    return lo == 1 && hi == nrow(c) ? c : c[lo:hi, :]
end

# The raw new time values. Both forms reject a textual column outright, as
# `resolvetime!` does: a String cannot be ordered against the window.
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

# The new column must live in the context's time type, as every source's does.
function converttimes(::Type{T}, times::AbstractVector, opname::String) where {T}
    eltype(times) <: T && return times
    return convert(Vector{T}, times)
end

# Function barrier: both vectors are concretely typed, so this compiles to a
# straight comparison loop. An explicit loop rather than `all(new .>= old)`,
# which would allocate a BitVector per chunk and lose the row index the message
# wants; `eachindex(old, new)` also asserts equal axes, which is what catches a
# `spec` function returning the wrong number of values.
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

# The Symbol form makes the named column the new :time — resolvetime!'s rename,
# with the twist that a :time column already exists and must go first, since
# `rename!` onto an existing name is an error. The renamed column keeps its own
# position, so the output schema is a fixed function of the input's. The Function
# form overwrites :time, which keeps its position for free. Both mutate the
# chunk's column index in place, which the owner may do (see DESIGN.md, "CSV
# output").
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
    return columnprojection(selectors, true, "selectcolumns")
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
    return columnprojection(selectors, false, "dropcolumns")
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
    return columnreorder(selectors)
end
reordercolumns(p::CausalPipeline, selectors...) = reordercolumns(selectors...)(p)

# Both transforms are the same chunkmap over a per-run resolution cache; they
# differ only in which side of the match survives.
function columnprojection(selectors::Tuple, selecting::Bool, opname::String)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            selection = ColumnSelection()
            return chunkmap(
                c -> projectchunk(selection, selectors, c, selecting, opname),
                p.run(ctx),
            )
        end
    end
end

# The resolved column list, cached against the schema it was resolved from:
# `keep === nothing` means every column survives and the chunk passes through
# untouched. Per-run mutable state lives here rather than in reassigned
# closure captures (which get boxed).
mutable struct ColumnSelection
    lastnames::Union{Nothing,Vector{String}}
    keep::Union{Nothing,Vector{Symbol}}
end
ColumnSelection() = ColumnSelection(nothing, nothing)

function projectchunk(selection::ColumnSelection, selectors::Tuple,
    c::DataFrame, selecting::Bool, opname::String)
    keep = resolvecolumns!(selection, selectors, c, selecting, opname)
    # The chunk is owned, so the projection can share its columns.
    return keep === nothing ? c : c[!, keep]
end

# Running the selectors over every column of every chunk is wasted work when
# the schema never moves — which is the norm — so the resolution is memoized
# against the names it was derived from. Re-resolving when they differ keeps
# the validation per-chunk-strict rather than first-chunk-only, since the
# trusted load/stream path does not itself re-check schema equality.
function resolvecolumns!(selection::ColumnSelection, selectors::Tuple,
    c::DataFrame, selecting::Bool, opname::String)
    cols = names(c)
    selection.lastnames == cols && return selection.keep
    selection.keep = keptcolumns(selectors, cols, selecting, opname)
    selection.lastnames = cols
    return selection.keep
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

# The projections' chunkmap-over-a-resolution-cache shape, differing only in
# what the resolution computes: a permutation of every column rather than a
# subset of them.
function columnreorder(selectors::Tuple)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            ordering = ColumnOrder()
            return chunkmap(c -> reorderchunk(ordering, selectors, c), p.run(ctx))
        end
    end
end

# The resolved column order, cached against the schema it was resolved from;
# `order === nothing` means the chunk is already in it and passes through
# untouched. Per-run mutable state, as in `ColumnSelection`.
mutable struct ColumnOrder
    lastnames::Union{Nothing,Vector{String}}
    order::Union{Nothing,Vector{Symbol}}
end
ColumnOrder() = ColumnOrder(nothing, nothing)

function reorderchunk(ordering::ColumnOrder, selectors::Tuple, c::DataFrame)
    order = resolveorder!(ordering, selectors, c)
    # The chunk is owned, so the reindex can share its columns.
    return order === nothing ? c : c[!, order]
end

# Memoized exactly as `resolvecolumns!` is, and for the same two reasons: the
# selectors are wasted work on a schema that never moves, and re-resolving when
# it does keeps the validation per-chunk-strict.
function resolveorder!(ordering::ColumnOrder, selectors::Tuple, c::DataFrame)
    cols = names(c)
    ordering.lastnames == cols && return ordering.order
    ordering.order = orderedcolumns(selectors, cols)
    ordering.lastnames = cols
    return ordering.order
end

# The column order to impose, or `nothing` when the chunk already has it.
# `placed` tracks by column position rather than by name: no hashing, and no
# second pass to subtract the columns the selectors claimed.
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

# Numbers and Chars iterate as scalars in Base, so they would recurse forever
# through the collection fallback below rather than being rejected by it.
const ScalarSelector = Union{Number,Char}

selectorerror(x) = throw(
    ArgumentError("invalid column selector of type $(typeof(x)): expected a \
        name, a Regex, a predicate, or a collection of those"))

checkselector(selectors) =
    applicable(iterate, selectors) || selectorerror(selectors)

# Does a selector spec match this column name? The leaf methods come first so
# a predicate (callable) and a collection (iterable) can never be confused;
# anything else must be iterable, and is matched recursively.
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

# Walk every leaf of a selector spec, in order — unlike `foreachliteral`, which
# visits only the name leaves. `reordercolumns` orders by the selectors, so it
# needs the regex and predicate ones too, and needs them in the order written.
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

# Eager validation: at least one selector, every leaf usable, and — for
# dropcolumns — no attempt to drop the time column every frame must have.
function checkselectors(selectors::Tuple, opname::String, dropping::Bool)
    isempty(selectors) &&
        throw(ArgumentError("$opname requires at least one column selector"))
    foreachliteral(selectors) do n
        dropping && n == "time" &&
            throw(ArgumentError("$opname may not drop the time column"))
    end
    return nothing
end
