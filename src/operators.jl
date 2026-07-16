# Sources return a CausalPipeline; transforms are curried and return a
# CausalPipeline -> CausalPipeline function, so both chain with |>.

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
        T = timetype(ctx)
        chunks = nothing        # file-chunk iterator, created on first pull
        state = nothing         # its iteration state
        started = false
        prevtime = nothing      # last raw time seen, for cross-chunk sortedness
        done = false
        return ChunkSource() do
            done && return nothing
            if chunks === nothing
                ntasks = max(1, Int(cld(filesize(path), chunkbytes)))
                # CSV.Chunks refuses files it cannot split (ntasks == 1, or
                # too few rows to justify it); such a file fits in one chunk,
                # so read it whole.
                chunks = if ntasks == 1
                    [CSV.read(path, DataFrame)]
                else
                    try
                        CSV.Chunks(path; ntasks = ntasks)
                    catch e
                        e isa ArgumentError ? [CSV.read(path, DataFrame)] :
                            rethrow()
                    end
                end
            end
            while true
                next = started ? iterate(chunks, state) : iterate(chunks)
                next === nothing && (done = true; return nothing)
                filechunk, state = next
                started = true
                # CSV.Chunks infers column types per chunk; a column's eltype
                # may differ between chunks, which load's vcat promotes.
                df = filechunk isa DataFrame ? filechunk : DataFrame(filechunk)
                "time" in names(df) ||
                    throw(ArgumentError("CSV file $path has no time column"))
                issorted(df.time) ||
                    throw(ArgumentError("time column in $path is not non-decreasing"))
                if nrow(df) > 0
                    prevtime !== nothing && first(df.time) < prevtime &&
                        throw(ArgumentError(
                            "time column in $path is not non-decreasing"))
                    prevtime = last(df.time)
                end
                lo = searchsortedfirst(df.time, ctx.start)
                hi = searchsortedfirst(df.time, ctx.stop) - 1
                hi < nrow(df) && (done = true)   # saw a time >= stop
                clipped = df[lo:hi, :]
                clipped[!, :time] = convert(Vector{T}, clipped.time)
                nrow(clipped) > 0 && return clipped
                done && return nothing
            end
        end
    end
end

"""
    filterrows(pred) -> (CausalPipeline -> CausalPipeline)

A transform keeping the rows where `pred(row)` is `true`. `pred` receives a
map-like row object supporting `row.name` and `row[:name]` access,
including `row.time`.
"""
function filterrows(pred)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> filter(pred, c), p.run(ctx))
        end
    end
end

"""
    addcolumns(f) -> (CausalPipeline -> CausalPipeline)

A transform adding columns computed row by row: `f(row)` must return a
`NamedTuple` mapping new column names to that row's values. The returned
tuple may not contain a `time` key. `f` receives the same map-like row
object as [`filterrows`](@ref).
"""
function addcolumns(f)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(p.run(ctx)) do c
                vals = [f(r) for r in eachrow(c)]
                first(vals) isa NamedTuple || throw(ArgumentError(
                    "addcolumns function must return a NamedTuple, got $(typeof(first(vals)))"))
                :time in keys(first(vals)) && throw(ArgumentError(
                    "addcolumns function may not return a time column"))
                return hcat(c, DataFrame(vals))
            end
        end
    end
end
