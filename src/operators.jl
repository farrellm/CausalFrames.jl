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
function renamecolumns!(df::DataFrame, m::AbstractDict)
    pairs = [k => m[k] for k in names(df) if haskey(m, k)]
    append!(
        pairs,
        [
            Symbol(k) => m[Symbol(k)] for k in names(df)
            if !haskey(m, k) && haskey(m, Symbol(k))
        ],
    )
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
    mask = rowmask(pred, Tables.columntable(c))
    return all(mask) ? c : c[mask, :]
end

# Function barrier: iterates concretely typed rows of the column table.
function rowmask(pred, nt::NamedTuple)
    mask = Vector{Bool}(undef, length(nt.time))
    for (i, row) in enumerate(Tables.rows(nt))
        mask[i] = pred(row)::Bool
    end
    return mask
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
