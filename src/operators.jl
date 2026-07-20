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
        renamecolumns!(df, p.rename)
        resolvetime!(df, p.time, p.path)
        issorted(df.time) ||
            throw(ArgumentError("time column in $(p.path) is not non-decreasing"))
        if nrow(df) > 0
            p.prevtime !== nothing && first(df.time) < p.prevtime &&
                throw(ArgumentError(
                    "time column in $(p.path) is not non-decreasing"))
            p.prevtime = last(df.time)
        end
        lo = searchsortedfirst(df.time, p.start)
        hi = searchsortedfirst(df.time, p.stop) - 1
        hi < nrow(df) && (p.done = true)   # saw a time >= stop
        # The file chunk is freshly materialized and owned, so a clip that
        # keeps every row needs no copy.
        clipped = lo == 1 && hi == nrow(df) ? df : df[lo:hi, :]
        clipped[!, :time] = convert(Vector{T}, clipped.time)
        nrow(clipped) > 0 && return clipped
        p.done && return nothing
    end
end

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
# was produced by a function.
function resolvetime!(df::DataFrame, time, path::String)
    if time isa Function
        df[!, :time] = maptime(time, Tables.columntable(df))
    else
        if time isa Symbol
            String(time) in names(df) ||
                throw(ArgumentError("CSV file $path has no column $(repr(time))"))
            time === :time || rename!(df, time => :time)
        end
        "time" in names(df) ||
            throw(ArgumentError("CSV file $path has no time column"))
        eltype(df.time) <: AbstractString &&
            throw(ArgumentError("time column in $path needs a concrete type \
                via `types`, or a `time` function to produce it"))
    end
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
