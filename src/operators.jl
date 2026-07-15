# Sources return a CausalPipeline; transforms are curried and return a
# CausalPipeline -> CausalPipeline function, so both chain with |>.

"""
    emptyframe() -> CausalPipeline

A source that always produces a frame with zero rows and only a `:time`
column.
"""
function emptyframe()
    return CausalPipeline() do ctx::Context
        T = timetype(ctx)
        return CausalFrame(ctx, DataFrame(time = T[]))
    end
end

"""
    clock(interval) -> CausalPipeline

A source producing one row per `interval` at times
`start, start + interval, ...` while `< stop`, with no columns other than
`:time`. `interval` may be anything addable to the context's time type
(e.g. a `Dates.Period` for `DateTime`, a number for numeric time).
"""
function clock(interval)
    return CausalPipeline() do ctx::Context
        T = timetype(ctx)
        ctx.start + interval > ctx.start ||
            throw(ArgumentError("clock interval must be positive, got $interval"))
        times = T[]
        t = ctx.start
        while t < ctx.stop
            push!(times, t)
            t += interval
        end
        return CausalFrame(ctx, DataFrame(time = times))
    end
end

"""
    readcsv(path) -> CausalPipeline

A source that reads the CSV file at `path`, which must contain a `time`
column sorted in non-decreasing order, and clips it to the context's
half-open interval `[start, stop)`. The `time` column is converted to the
context's time type.
"""
function readcsv(path::AbstractString)
    return CausalPipeline() do ctx::Context
        T = timetype(ctx)
        df = CSV.read(path, DataFrame)
        "time" in names(df) ||
            throw(ArgumentError("CSV file $path has no time column"))
        issorted(df.time) ||
            throw(ArgumentError("time column in $path is not non-decreasing"))
        lo = searchsortedfirst(df.time, ctx.start)
        hi = searchsortedfirst(df.time, ctx.stop) - 1
        clipped = df[lo:hi, :]
        clipped[!, :time] = convert(Vector{T}, clipped.time)
        return CausalFrame(ctx, clipped)
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
            frame = load(ctx, p)
            chunks = DataFrame[filter(pred, c) for c in frame.chunks]
            return CausalFrame(ctx, chunks)
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
            frame = load(ctx, p)
            chunks = map(frame.chunks) do c
                vals = [f(r) for r in eachrow(c)]
                first(vals) isa NamedTuple || throw(ArgumentError(
                    "addcolumns function must return a NamedTuple, got $(typeof(first(vals)))"))
                :time in keys(first(vals)) && throw(ArgumentError(
                    "addcolumns function may not return a time column"))
                return hcat(c, DataFrame(vals))
            end
            return CausalFrame(ctx, chunks)
        end
    end
end
