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
emptyframe() = CausalPipeline(ctx -> DataFrame[])

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
    readcsv(path; chunkbytes = 4 * 1024 * 1024) -> CausalPipeline

A source that reads the CSV file at `path`, which must contain a `time`
column sorted in non-decreasing order, and clips it to the context's
half-open interval `[start, stop)`. The `time` column is converted to the
context's time type.

The file is read incrementally in chunks of roughly `chunkbytes` bytes —
never all at once — and reading stops as soon as a time `>= stop` is seen.
Consequently a sortedness violation is only detected when the offending
chunk is actually read.
"""
function readcsv(path::AbstractString; chunkbytes::Integer = 4 * 1024 * 1024)
    chunkbytes > 0 ||
        throw(ArgumentError("readcsv chunkbytes must be positive, got $chunkbytes"))
    return CausalPipeline() do ctx::Context
        return ChunkSource(CSVProducer{timetype(ctx)}(String(path), Int(chunkbytes),
                                                      ctx.start, ctx.stop))
    end
end

# The stateful producer behind readcsv's ChunkSource. The pull-to-pull state
# lives in fields rather than captured locals (captured variables that are
# reassigned get boxed). The dynamically typed fields are per-chunk setup
# state, not per-row state.
mutable struct CSVProducer{T}
    const path::String
    const chunkbytes::Int
    const start::T
    const stop::T
    chunks::Any        # file-chunk iterator, created on first pull
    state::Any         # its iteration state
    started::Bool
    prevtime::Any      # last raw time seen, for cross-chunk sortedness
    done::Bool
    CSVProducer{T}(path, chunkbytes, start, stop) where {T} =
        new{T}(path, chunkbytes, start, stop, nothing, nothing, false, nothing,
               false)
end

# CSV.Chunks refuses files it cannot split (ntasks == 1, or too few rows to
# justify it); such a file fits in one chunk, so read it whole.
function csvchunks(path::String, chunkbytes::Int)
    ntasks = max(1, Int(cld(filesize(path), chunkbytes)))
    ntasks == 1 && return [CSV.read(path, DataFrame)]
    try
        return CSV.Chunks(path; ntasks = ntasks)
    catch e
        e isa ArgumentError ? [CSV.read(path, DataFrame)] : rethrow()
    end
end

function (p::CSVProducer{T})() where {T}
    p.done && return nothing
    p.chunks === nothing && (p.chunks = csvchunks(p.path, p.chunkbytes))
    while true
        next = p.started ? iterate(p.chunks, p.state) : iterate(p.chunks)
        next === nothing && (p.done = true; return nothing)
        filechunk, p.state = next
        p.started = true
        # CSV.Chunks infers column types per chunk; a column's eltype may
        # differ between chunks, which load's vcat promotes.
        df = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)
        "time" in names(df) ||
            throw(ArgumentError("CSV file $(p.path) has no time column"))
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
    first(vals) isa NamedTuple || throw(ArgumentError(
        "addcolumns function must return a NamedTuple, got $(typeof(first(vals)))"))
    :time in keys(first(vals)) && throw(ArgumentError(
        "addcolumns function may not return a time column"))
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, DataFrame(vals); copycols = false)
end

# Function barrier: with concretely typed rows the comprehension infers, so
# the collected values have a concrete NamedTuple eltype and DataFrame builds
# typed columns from them directly.
rowvalues(f, nt::NamedTuple) = [f(row) for row in Tables.rows(nt)]
