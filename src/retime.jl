# The transforms that move rows in time or cut the stream: the causal time
# shift (`lag`), the context-widening `warmup`, the truncating `head` and the
# causal retiming (`settime`).

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
    return pipetransform(; contexts = ctx -> lagcontext(ctx, offset)) do upstream, ctx
        return chunkmap(c -> shiftchunk!(c, offset), upstream)
    end
end
lag(p::CausalPipeline, offset) = lag(offset)(p)

# The input window: the whole window slid back by `offset`, so the shift lands
# the output in [start, stop). A negative offset (acausal) is rejected by
# `widenstart`'s probe.
lagcontext(ctx::Context, offset) =
    Context(widenstart(ctx, offset, "lag offset").start, ctx.stop - offset)

# Shift the owned chunk's time column by a constant, keeping its order and
# position. Shared with `Acausal.lead`, which passes a negative `delta`.
shiftchunk!(c::DataFrame, delta) = (c[!, :time] = shifttime(c.time, delta); c)

# Function barrier: the broadcast specializes on the concretely typed column.
shifttime(times::AbstractVector, delta) = times .+ delta

"""
    warmup(lookback, f) -> (CausalPipeline -> CausalPipeline)
    warmup(p::CausalPipeline, lookback, f) -> CausalPipeline

A transform running `f(p)` over `[start - lookback, stop)` and dropping its
output rows before `start`, so a stateful transform enters the window already
warmed up instead of restarting at its first row. `f` sees the lead-in as input:
[`summarize`](@ref) folds it and [`head`](@ref) counts it.

If `lookback` covers everything `f` remembers, loading `[a, b)` and `[b, c)` and
concatenating the results equals loading `[a, c)`. That holds exactly for finite
memory (time windows, `forwardfill` with a `tolerance`, and [`Bars`](@ref) when
`lookback` spans the bars), and only to decay tolerance for a recursive state.
It never holds for a path-dependent state, such as a cumulative
[`Sum`](@ref) in [`addsummarycolumns`](@ref), nor for output aligned to the
context's start, such as [`clock`](@ref) ticks.

Warm-ups compose by adding their lookbacks: for non-negative `x` and `y`,
`warmup(x, warmup(y, f))` equals `warmup(x + y, f)` whenever
`(start - x) - y == start - (x + y)`, which floating-point rounding and
calendar periods near a month's end can break.

# Arguments
- `lookback`: how far before `start` `f` runs, in a type that can be subtracted
  from the time type (a `Dates.Period`, a number). Must be non-negative, checked
  when the pipeline runs; `0` and `nothing` are the identity.
- `f`: a function mapping a `CausalPipeline` to a `CausalPipeline`, such as a
  transform or a chain of them. Called once, when `warmup` is applied; any other
  return value is an `ArgumentError`.

```jldoctest
df = DataFrame(time = [1, 4, 6], bid = [1.0, missing, missing])
p = readtable(df) |> warmup(5, forwardfill(:bid))
DataFrame(load(Context(3, 10), p))

# output

2×2 DataFrame
 Row │ time   bid
     │ Int64  Float64?
─────┼─────────────────
   1 │     4       1.0
   2 │     6       1.0
```
"""
function warmup(lookback, f)
    return function (@nospecialize(p::CausalPipeline),)
        q = f(p)
        q isa CausalPipeline || throw(ArgumentError(
            "warmup f must return a CausalPipeline, got $(typeof(q))"))
        # As free of the context as `q` is: dropping the lead-in is.
        return pipenode((q,);
            contexts = ctx -> widenstart(ctx, lookback, "warmup lookback")) do ctx, ins
            st = LeadInDrop(ctx.start)
            return chunkmap(c -> dropleadin!(st, c), openinput(ins, 1))
        end
    end
end
warmup(p::CausalPipeline, lookback, f) = warmup(lookback, f)(p)

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
`[b, c)` can yield `2n` rows where `[a, c)` yields `n`. For the same reason a
pipeline ending in `head` read over two contexts with different starts runs
once for each (see [`sharing`](@ref)).

Put writers downstream of `head`, never upstream (`p |> head(n) |>
writecsv(path)`): a writer finalizes its file only when its input is exhausted,
which `head` prevents.
"""
function head(n::Integer)
    checknonnegative(n, "head n")
    limit = Int(n)
    # The count starts at the run's start.
    return pipetransform(; startfree = false) do upstream, ctx
        return ChunkSource(HeadProducer(upstream, limit))
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
    if p.remaining <= 0
        abandon!(p.upstream)
        return nothing
    end
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
    abandon!(p.upstream)
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
[`Acausal.settime`](@ref CausalFrames.Acausal.settime)), a new time column
that decreases within or across chunks, a textual new time, and a `missing` one
(drop such rows first with [`filterrows`](@ref)).

The input is not widened: it runs over `[start, stop)`, so a row outside the
window is never seen, even if `spec` would move it inside. Hence loading
`[a, c)` need not equal loading `[a, b)` and `[b, c)`. `settime(:time)` changes
no values, but still drops rows at `stop`.
"""
function settime(spec)
    checktimespec(spec, "settime")
    # A row before `start` moved into the window is never seen.
    return pipetransform(; startfree = false) do upstream, ctx
        st = SetTimeState{timetype(ctx)}()
        return chunkmap(
            c -> settimechunk!(st, spec, c, ctx.start, ctx.stop, true, "settime"),
            upstream,
        )
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
    new = converttimes(T, newtimes(spec, c, opname))
    causal && checkforward(old, new, opname)
    issorted(new) ||
        throw(ArgumentError("$opname produced a time column that is not non-decreasing"))
    st.prevtime === nothing || st.prevtime <= first(new) ||
        throw(
            ArgumentError("$opname produced a time column that is not non-decreasing \
                across chunk boundaries"),
        )
    st.prevtime = last(new)          # pre-clip, as clipchunk! carries it
    lo, hi = windowbounds(new, false, start, stop)
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

# The raw new time values, refused if textual or if any is `missing`.
function checktimevalues(v::AbstractVector, opname::String)
    istextual(v) && throw(
        ArgumentError("$opname produced a textual time column (element type \
            $(eltype(v))); parse it to an ordered type first"),
    )
    Missing <: eltype(v) && any(ismissing, v) &&
        throw(
            ArgumentError("$opname produced a missing time; drop those rows first \
            with `filterrows`"),
        )
    return v
end

# The new column takes the context's time type, as every source's does.
function converttimes(::Type{T}, times::AbstractVector) where {T}
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
# the owned chunk's column index (the `writecsv` hand-off argument).
function settimecolumn!(c::DataFrame, spec::Symbol, new::AbstractVector)
    if spec !== :time
        select!(c, Not(:time))
        rename!(c, spec => :time)
    end
    c[!, :time] = new
    return c
end
settimecolumn!(c::DataFrame, ::Function, new::AbstractVector) = (c[!, :time] = new; c)
