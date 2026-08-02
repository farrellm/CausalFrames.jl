# Sources return a CausalPipeline; transforms are curried and return a
# CausalPipeline -> CausalPipeline function, so both chain with |>.
#
# Row functions (filterrows' pred, addcolumns' f) receive the concretely
# typed rows of a Tables.columntable, iterated behind a function barrier —
# never DataFrameRows, whose column accesses are type-unstable.

"""
    emptyframe() -> CausalPipeline

A source that always produces zero rows (loading it yields a frame with only
a `:time` column).
"""
emptyframe() = CausalPipeline(ctx -> ChunkSource(() -> nothing))

"""
    concatenate(ps::CausalPipeline...) -> CausalPipeline

A source running the given pipelines one after another over the same context
and emitting their chunks end to end — the time-wise concatenation of their
outputs. There is no interleaving and no merging: the pipelines must be passed
in time order and must produce identical column names.

Every pipeline is evaluated over the whole context `[start, stop)` and clips
itself, so keeping their windows from overlapping is the caller's business.
Each is run only once the previous one is exhausted, which keeps the chain as
lazy as its parts — a chain of file sources holds one file open at a time.

Both requirements are checked as the chunks flow by: an `ArgumentError` is
thrown when a chunk's column names differ from the first chunk's, or when a
chunk's first time precedes the last time already emitted (equal times across
a boundary are fine). Element *types* may differ between pipelines, as they
may between the chunks of one pipeline — `DataFrame(frame)` promotes on
concatenation.

`concatenate()` with no pipelines is [`emptyframe`](@ref), the identity of
concatenation.

```julia
concatenate(readcsv("jan.csv"; types = tt), readcsv("feb.csv"; types = tt)) |>
    filterrows(r -> r.bid > 0)
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

A source producing one row per `interval` at times
`start, start + interval, ...` while `< stop`, with no columns other than
`:time`. `interval` may be anything addable to the context's time type
(e.g. a `Dates.Period` for `DateTime`, a number for numeric time). Ticks are
generated lazily in chunks of `batchsize` rows.
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
            delim = nothing, chunkbytes = 4 * 1024 * 1024) -> CausalPipeline

A source that reads the CSV file at `path` and clips it to the context's
half-open interval `[start, stop)`. Every column is read as `String` — types
are **not** inferred — unless `types` opts a column into a concrete type.

The resulting time column, whatever its source, is materialized as `:time`,
must be sorted in non-decreasing order, and is converted to the context's
time type. It is chosen by `time`:

- `time = nothing` (default): the column already named `:time`.
- `time = :name` (a `Symbol`): the column named `:name` (after `rename`),
  renamed to `:time`.
- `time = f` (a function): `f(row)` is called per row to compute the time
  value, producing the `:time` column (any existing `:time` is overwritten).

Since a `String` time column cannot be ordered against the numeric window,
the time column must be typed unless produced by a function: it is an
`ArgumentError` if `time` is not a function and `types` gives no concrete
type for the time column.

Keyword arguments:

- `types`: which columns to give a concrete type (everything else stays
  `String`), as a `Dict`/`Vector`/function over the file's *original* column
  names or indices — it is applied while parsing, so it is keyed by the names
  in the file, before any `rename`.
- `rename`: an `AbstractDict`/map (over original names) or a `name -> name`
  function applied to the column names **after** typing but **before** `time`
  is resolved.
- `delim`: the field delimiter, passed through to `CSV.Chunks` (a `Char` or
  `String`); defaults to CSV.jl's own detection.
- `chunkbytes`: the file is read incrementally in chunks of roughly this many
  bytes — never all at once — and reading stops as soon as a time `>= stop`
  is seen. Consequently a sortedness violation is only detected when the
  offending chunk is actually read.
"""
function readcsv(path::AbstractString; types = nothing, time = nothing,
    rename = nothing, delim = nothing, chunkbytes::Integer = 4 * 1024 * 1024)
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
                ctx.start, ctx.stop, types, time, rename, delim),
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
    chunks::Any         # file-chunk iterator, created on first pull
    state::Any          # its iteration state
    started::Bool
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool
    CSVProducer{T}(path, chunkbytes, start, stop, types, time, rename,
        delim) where {T} =
        new{T}(path, chunkbytes, start, stop, types, time, rename, delim,
            nothing, nothing, false, nothing, false)
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
    opts = (; types = typesfunction(types), stringtype = String, delim = delim)
    ntasks = max(1, Int(cld(filesize(path), chunkbytes)))
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
            "CSV file", p.prevtime, p.start, p.stop)
        sawstop && (p.done = true)
        nrow(clipped) > 0 && return clipped
        p.done && return nothing
    end
end

# Shared by readcsv and readparquet: rename the columns, materialize `:time`,
# check sortedness within the chunk and against the last time of the previous
# one, clip to [start, stop), and convert `:time` to the context's time type.
# `what` names the format in error messages. Returns the clipped chunk (which
# may have no rows), whether a time >= stop was seen (the source is then done),
# and the last raw time of this chunk, to be carried to the next call.
function clipchunk!(df::DataFrame, time, rename, path::String, what::String,
    prevtime, start::T, stop::T) where {T}
    renamecolumns!(df, rename)
    resolvetime!(df, time, path, what)
    issorted(df.time) ||
        throw(ArgumentError("time column in $path is not non-decreasing"))
    if nrow(df) > 0
        prevtime !== nothing && first(df.time) < prevtime &&
            throw(ArgumentError("time column in $path is not non-decreasing"))
        prevtime = last(df.time)
    end
    lo = searchsortedfirst(df.time, start)
    hi = searchsortedfirst(df.time, stop) - 1
    sawstop = hi < nrow(df)   # saw a time >= stop
    # The chunk is freshly materialized and owned, so a clip that keeps every
    # row needs no copy.
    clipped = lo == 1 && hi == nrow(df) ? df : df[lo:hi, :]
    clipped[!, :time] = convert(Vector{T}, clipped.time)
    return (clipped, sawstop, prevtime)
end

# A time column that arrived as text cannot be ordered against the window. Only
# CSV has a `types` knob to point the user at; parquet carries its own types, so
# there the only way out is a `time` function.
textualtime(path::String, what::String) =
    what == "CSV file" ?
    "time column in $path needs a concrete type via `types`, or a `time` \
    function to produce it" :
    "time column in $path is textual; use a `time` function to produce a \
    usable time"

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

# Materialize the `:time` column and check it is usable (non-String) unless it
# was produced by a function. `what` names the file format for error messages;
# only CSV has a `types` knob to point the user at.
function resolvetime!(df::DataFrame, time, path::String, what::String)
    if time isa Function
        df[!, :time] = maptime(time, Tables.columntable(df))
    else
        if time isa Symbol
            String(time) in names(df) ||
                throw(ArgumentError("$what $path has no column $(repr(time))"))
            time === :time || rename!(df, time => :time)
        end
        "time" in names(df) ||
            throw(ArgumentError("$what $path has no time column"))
        eltype(df.time) <: AbstractString &&
            throw(ArgumentError(textualtime(path, what)))
    end
    return nothing
end

"""
    writecsv(path; queue = 1, kwargs...) -> (CausalPipeline -> CausalPipeline)
    writecsv(p::CausalPipeline, path; ...) -> CausalPipeline

A transparent pass-through transform that writes the stream to the CSV file
at `path` as it flows by, yielding every chunk downstream unchanged. Nothing
is buffered: each chunk is written and flushed as it is produced, so the file
grows while the pipeline is still running.

Writing happens on a background task fed by a bounded queue, so the pipeline
does not block on disk I/O — only if the writer falls more than `queue`
chunks behind, plus once at the end to join it. `queue = 0` makes each
hand-off a rendezvous.

The file is truncated when the run starts and finalized when the stream is
*exhausted* — by [`load`](@ref), [`scan`](@ref), or a fully drained
[`stream`](@ref). Abandoning a `stream` part-way leaves the last chunks
unwritten; use [`scan`](@ref) when the file is all you want:

```julia
scan(ctx, readcsv("ticks.csv"; types = tt) |>
          addcolumns(r -> (; mid = (r.bid + r.ask) / 2)) |>
          writecsv("mids.csv"))
```

A stream with no rows at all yields an empty file. Keyword arguments are
passed through to `CSV.write` (`delim`, `missingstring`, `dateformat`,
`quotestrings`, `bufsize`, …), except for `append`, `header`, `writeheader`,
`partition` and `compress`, which this transform controls itself — passing
one is an `ArgumentError`.

The curried form composes with `|>`; the uncurried form applies directly, so
`writecsv(p, path)` is equivalent to `p |> writecsv(path)`.
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

A transform keeping the rows where `pred(row)` is `true`. `pred` receives a
map-like row object supporting `row.name` and `row[:name]` access,
including `row.time`.

The curried form composes with `|>`; the uncurried form applies directly, so
`filterrows(p, pred)` is equivalent to `p |> filterrows(pred)`.
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

A transform adding columns computed row by row: `f(row)` must return a
`NamedTuple` mapping new column names to that row's values. The returned
tuple may not contain a `time` key. `f` receives the same map-like row
object as [`filterrows`](@ref).

The curried form composes with `|>`; the uncurried form applies directly, so
`addcolumns(p, f)` is equivalent to `p |> addcolumns(f)`.
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

A transform shifting every row `offset` later in time (`time -> time +
offset`), so that the value observed at time `t` is the one the input carried
at `t - offset` — the lagged, backward-looking view. Only the `:time` column
changes; all other columns pass through unchanged.

`offset` must be non-negative (a negative shift would look into the future and
break causality — see [`lead`](@ref CausalFrames.Acausal.lead) for that); an
`ArgumentError` is thrown when the pipeline runs otherwise, and `offset` `== 0`
is the identity. The time type must support adding and subtracting the offset
(numbers and `Dates` types do): the upstream pipeline is run over the window
`[start - offset, stop - offset)` so the shifted output covers `[start, stop)`.

The curried form composes with `|>`; the uncurried form applies directly, so
`lag(p, offset)` is equivalent to `p |> lag(offset)`.
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

A transform emitting the first `n` rows of the stream and then stopping. It
stops for real: once `n` rows have gone out the upstream pipeline is never
pulled again, so `readcsv(path; types) |> head(10)` reads one file chunk and no
more. `n` must be non-negative (an `ArgumentError` at construction otherwise),
and `head(0)` is the empty stream — the upstream pipeline is built but never
iterated.

`head` is causal — rows pass through unchanged and in order — but **stateful**:
its row budget spans the whole window. Concatenating the frames of
[`stream`](@ref) therefore still equals [`load`](@ref) of the same window, while
the chunk-concatenation property over *split* contexts does not hold, since
`head(n)` over `[a, b)` and over `[b, c)` yields up to `2n` rows where the whole
window yields `n`.

Do not put a sink upstream of `head`: [`writecsv`](@ref) and
[`writeparquet`](@ref) finalize when the stream is *exhausted*, which `head`
prevents, leaving the file unfinished and its writer task unjoined. Truncate
first — `p |> head(n) |> writecsv(path)`.

The curried form composes with `|>`; the uncurried form applies directly, so
`head(p, n)` is equivalent to `p |> head(n)`.
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

A transform recomputing the `:time` column. `spec` is either

- a `Symbol` — the named column *becomes* `:time`, taking over the position it
  already occupied, and the old `:time` column disappears ([`readcsv`](@ref)'s
  `time = :name` rename, applied mid-stream); or
- a per-row function — `spec(row)` is called for each row, receiving the same
  map-like row object as [`addcolumns`](@ref), and its result overwrites
  `:time` in place, keeping that column's position.

Either way the resulting column is converted to the context's time type and the
chunk is re-clipped to the half-open interval `[start, stop)`. All other columns
pass through unchanged.

The transform is **causal**, and enforces it. Three independent checks, each an
`ArgumentError` when the pipeline runs:

- every row's new time is at least its old one (use
  [`CausalFrames.Acausal.settime`](@ref) to move rows earlier);
- the resulting column is non-decreasing within the chunk;
- and it does not step back across a chunk boundary.

None implies another: a forward-only map can still reorder rows, and an ordered
map can move every row back to `start`.

The context is **not widened**. [`lag`](@ref) slides its upstream window because
its shift is a constant known before any data is read; `settime`'s is per-row
and data-dependent, so upstream still runs over `[start, stop)` and only rows
*already* in the window can be retimed — a row before `start` that `spec` would
move into the window is never seen. For the same reason `settime` does not have
the chunk-concatenation property: loading `[a, c)` is not the concatenation of
loading `[a, b)` and `[b, c)`, since a row retimed across `b` is dropped by the
first half and never offered to the second. Streaming still equals loading.

`settime(:time)` is legal and leaves the values alone, but it still re-clips to
`[start, stop)` — a row sitting exactly at `stop`, which frames tolerate and
[`summarize`](@ref) emits, is dropped.

The curried form composes with `|>`; the uncurried form applies directly, so
`settime(p, spec)` is equivalent to `p |> settime(spec)`.
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

# Eager validation, shared by the two variants: a column name or a per-row
# function, nothing else. A CausalPipeline lands here too, which is what turns a
# mistyped `settime(p)` into a message rather than a MethodError deep in a chunk.
checktimespec(::Symbol, ::String) = nothing
checktimespec(::Function, ::String) = nothing
checktimespec(x, opname::String) = throw(
    ArgumentError(
        "$opname spec must be a column name (Symbol) or a per-row function, \
        got $(typeof(x))"),
)

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
                $(old[i]) to $(new[i]); use CausalFrames.Acausal.settime for that"),
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

A transform keeping only the selected columns, in the input's own column
order. Each selector is a column name (a `Symbol` or `AbstractString`), a
`Regex` matched against the column name, a predicate called with the column
name as a `String`, or — recursively — any collection of those; a column is
kept when it matches any of them.

`:time` is always kept, whether or not it is selected. Naming a column that
the data does not have is an `ArgumentError`; a `Regex` or predicate matching
nothing is not.

The curried form composes with `|>`; the uncurried form applies directly, so
`selectcolumns(p, sel)` is equivalent to `p |> selectcolumns(sel)`.

```julia
p |> selectcolumns(:bid, :ask)
p |> selectcolumns(r"^px_", startswith("qty"))
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

A transform dropping the selected columns and keeping the rest, in the
input's own column order. Selectors take the same forms as for
[`selectcolumns`](@ref), and a column is dropped when it matches any of them.

`:time` is never dropped — a `Regex` or predicate matching it is ignored, and
naming it outright is an `ArgumentError`, since every frame must have a time
column. Naming a column that the data does not have is an `ArgumentError`
too.

The curried form composes with `|>`; the uncurried form applies directly, so
`dropcolumns(p, sel)` is equivalent to `p |> dropcolumns(sel)`.
"""
function dropcolumns(selectors...)
    checkselectors(selectors, "dropcolumns", true)
    return columnprojection(selectors, false, "dropcolumns")
end
dropcolumns(p::CausalPipeline, selectors...) = dropcolumns(selectors...)(p)

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
