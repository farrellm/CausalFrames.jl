# The basic sources: the empty stream, sequential concatenation and the
# clock. Sources return a CausalPipeline; transforms are curried and return a
# CausalPipeline -> CausalPipeline function, so both chain with |>.

"""
    emptyframe() -> CausalPipeline

A source producing no rows. Loading it gives a zero-row frame with only
`:time`. It is the identity of [`concatenate`](@ref) and [`merge`](@ref "merge").
"""
emptyframe() = pipesource(ctx -> ChunkSource(() -> nothing))

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
    pipenode((ctx, ins) -> ChunkSource(ConcatProducer(ins, length(ps))), ps)

# The stateful producer behind concatenate's ChunkSource. State lives in fields
# rather than captured locals, which would be boxed when reassigned. The untyped
# fields are touched once per chunk or per pipeline, never per row.
mutable struct ConcatProducer
    const ins::PipelineInputs
    const n::Int        # the number of pipelines
    index::Int          # the pipeline currently being drained
    cursor::Any         # a PullCursor over its chunks; nothing until reached
    names::Union{Nothing,Vector{String}}  # column names of the first chunk
    prevtime::Any       # last time emitted, for cross-pipeline ordering
    previndex::Int      # which pipeline emitted it, for the error message
    ConcatProducer(ins::PipelineInputs, n::Int) =
        new(ins, n, 1, nothing, nothing, nothing, 0)
end

function (p::ConcatProducer)()
    while p.index <= p.n
        p.cursor === nothing && (p.cursor = PullCursor(openinput(p.ins, p.index)))
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

# The pipeline being drained, and the ones never reached.
function abandon!(p::ConcatProducer)
    p.cursor === nothing || abandon!(p.cursor)
    for i in (p.cursor===nothing ? p.index : p.index+1):p.n
        abandoninput(p.ins, i)
    end
    p.index = p.n + 1
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
    checkpositive(batchsize, "clock batchsize")
    # Ticks are anchored to the run's start.
    return pipesource(; startfree = false) do ctx::Context
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
