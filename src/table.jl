# `readtable`, the source over in-memory data. Three paths share one clip rule
# (`windowbounds`) and differ in when the per-table work runs and in what is
# copied:
#
# - a generic Tables.jl table is resolved per run and per partition by a
#   CSVProducer-shaped `TableProducer`, copying only the in-window rows;
# - an AbstractDataFrame is resolved once, at construction, into a private
#   index over the caller's vectors, and every run takes the frame path;
# - a CausalFrame's chunks are clipped as they stand (`framepipeline`), a chunk
#   wholly inside the window handed on as a private index over its vectors.

"""
    readtable(table; time = nothing, checkorder = true, sort = false,
              closed = false) -> CausalPipeline
    readtable(frame::CausalFrame; closed = nothing, checkcontext = true)
        -> CausalPipeline

A source lifting in-memory data into a pipeline: any Tables.jl table — a
`DataFrame`, a `NamedTuple` of vectors, a vector of `NamedTuple`s, a loaded
[`CausalFrame`](@ref) — clipped to the context's half-open interval
`[start, stop)`, with `:time` converted to the context's time type.

The time column is chosen by `time`, as for [`readcsv`](@ref):

- `time = nothing` (default): the column named `:time`.
- `time = :name`: the column `:name`, renamed to `:time` where it stands (an
  `ArgumentError` if the table also has a `:time` column).
- `time = f` (a function): `f(row)` is called per row to compute `:time`,
  overwriting any existing `:time` column.

A textual time column cannot be ordered against the window and is an
`ArgumentError`.

Keyword arguments:

- `checkorder`: the time column must be non-decreasing, which is checked unless
  `checkorder = false` — the caller then vouches for the order, and an unsorted
  table clips to nonsense rather than to an error.
- `sort`: sort the rows by time first. The sort is stable, so rows sharing a
  timestamp keep their order; a table with several partitions is concatenated
  to sort it.
- `closed`: clip to the closed interval `[start, stop]` instead, keeping the
  rows at `stop` (which a frame tolerates).

Only the rows inside the window are copied, so a narrow window over a large
table costs the window, and a loaded frame never shares memory with `table`. A
table with several `Tables.partitions` is read one chunk per partition, and
reading stops at the first partition holding a time past the window.

A `DataFrame` (any `AbstractDataFrame`) is prepared once, when `readtable` is
called: the time column is resolved and its order checked (or sorted) there, so
those errors are raised by `readtable` itself, and each run only searches for the
window. `readtable` keeps a reference to the DataFrame rather than a copy, so
mutating its values while the pipeline is still in use is unsupported.

A `CausalFrame` is read back chunk by chunk, and a chunk lying wholly inside the
window is not copied at all. A frame knows its rows only over its own context,
so running over a context that is not within `context(frame)` is an
`ArgumentError` unless `checkcontext = false`, which clips the frame to it
anyway. `closed = nothing` (the frame default) keeps the rows at `stop` exactly
when the run's `stop` equals the frame's, so `load(context(frame),
readtable(frame))` reproduces a frame holding rows at its stop — a
[`summarize`](@ref) output, say; `true` or `false` forces the choice. `time`,
`checkorder` and `sort` are not accepted for a frame, whose `:time` is already
resolved and sorted.

```julia
df = DataFrame(ts = [3, 1, 2], bid = [1.0, 2.0, 3.0])
readtable(df; time = :ts, sort = true) |> filterrows(r -> r.bid > 1)
```
"""
function readtable(table; time = nothing, checkorder::Bool = true,
    sort::Bool = false, closed::Bool = false)
    checktabletimespec(time)
    Tables.istable(table) ||
        throw(ArgumentError("readtable: a $(typeof(table)) is not a Tables.jl table"))
    return CausalPipeline() do ctx::Context
        return ChunkSource(
            TableProducer{timetype(ctx)}(table, time, checkorder,
                sort, closed, ctx.start, ctx.stop),
        )
    end
end

function readtable(df::AbstractDataFrame; time = nothing, checkorder::Bool = true,
    sort::Bool = false, closed::Bool = false)
    checktabletimespec(time)
    # A private index over the caller's vectors: O(ncols), and the rename and
    # the :time assignment below never reach `df` itself.
    wrapped = DataFrame(df; copycols = false)
    times, source = tabletimes(wrapped, propertynames(wrapped), time)
    if source === nothing
        wrapped[!, :time] = times
    elseif source !== :time
        rename!(wrapped, source => :time)
    end
    # Until a sort has permuted the rows into copies of its own, the chunk
    # aliases the caller's vectors, and every run must slice rather than share.
    owned = false
    if sort
        if !issorted(times)
            wrapped = wrapped[stableperm(times), :]
            owned = true
        end
    elseif checkorder
        issorted(times) ||
            throw(ArgumentError("time column in table is not non-decreasing"))
    end
    chunks = nrow(wrapped) > 0 ? [wrapped] : DataFrame[]
    return framepipeline(chunks, nothing, closed, false, owned)
end

readtable(frame::CausalFrame; closed::Union{Nothing,Bool} = nothing,
    checkcontext::Bool = true) =
    framepipeline(frame.chunks, frame.context, closed, checkcontext, true)

checktabletimespec(::Nothing) = nothing
checktabletimespec(time) = checktimespec(time, "readtable time")

# The generic path's producer. The pull-to-pull state lives in fields rather
# than reassigned closure captures (which would be boxed); the dynamically
# typed fields are per-partition setup state, and the per-row work sits behind
# `tablerows`, `copytimes` and `maptime`.
mutable struct TableProducer{T,X}
    const table::X
    const time::Any      # Nothing | Symbol (column name) | Function (row -> time)
    const checkorder::Bool
    const sort::Bool
    const closed::Bool
    const start::T
    const stop::T
    parts::Any           # the partition iterator, created on the first pull
    state::Any           # its iteration state
    started::Bool
    prevtime::Any        # last time of the previous partition, for its order
    done::Bool
    TableProducer{T}(table::X, time, checkorder, sort, closed, start,
        stop) where {T,X} =
        new{T,X}(table, time, checkorder, sort, closed, start, stop, nothing,
            nothing, false, nothing, false)
end

function (p::TableProducer{T})() where {T}
    p.done && return nothing
    p.parts === nothing && (p.parts = tablepartitions(p.table, p.sort))
    while true
        next = p.started ? iterate(p.parts, p.state) : iterate(p.parts)
        if next === nothing
            p.done = true
            return nothing
        end
        part, p.state = next
        p.started = true
        chunk, sawstop, p.prevtime = tablechunk(part, p.time, p.checkorder,
            p.sort, p.closed, p.prevtime, p.start, p.stop)
        sawstop && (p.done = true)
        chunk === nothing || return chunk
        p.done && return nothing
    end
end

# A sort has to see the whole table, so a partitioned table is concatenated to
# sort it; otherwise the partitions are the chunks.
function tablepartitions(table, sort::Bool)
    parts = Tables.partitions(table)
    sort || return parts
    collected = collect(parts)
    length(collected) <= 1 && return collected
    return (reduce(vcat, [DataFrame(x; copycols = false) for x in collected]),)
end

# One partition as a chunk holding only its in-window rows. Returns the chunk
# (nothing when no row is in the window), whether a time past the window was
# seen (the source is then done), and the last time, carried to the next
# partition's order check.
function tablechunk(part, time, checkorder::Bool, sort::Bool, closed::Bool,
    prevtime, start::T, stop::T) where {T}
    cols = Tables.columns(part)
    colnames = Tables.columnnames(cols)
    times, source = tabletimes(cols, colnames, time)
    rows, sawstop, prevtime =
        tablerows(times, checkorder, sort, closed, prevtime, start, stop)
    isempty(rows) && return (nothing, sawstop, prevtime)
    chunk = assemblechunk(cols, colnames, source, copytimes(T, times, rows), rows)
    return (chunk, sawstop, prevtime)
end

# The raw time values, and the column they came from (`nothing` for a function,
# whose values become a `:time` column). `cols` is a Tables.jl column table.
tabletimes(cols, colnames, time::Function) =
    (checktabletimes(maptime(time, Tables.columntable(cols))), nothing)
function tabletimes(cols, colnames, time::Union{Nothing,Symbol})
    name = something(time, :time)
    name in colnames || throw(
        ArgumentError(
            time === nothing ? "table has no time column; choose one with `time`" :
            "table has no column $(repr(name))"),
    )
    name === :time || !(:time in colnames) ||
        throw(
            ArgumentError("table has both a :time column and the time column \
            $(repr(name)); drop one of them first"),
        )
    return (checktabletimes(Tables.getcolumn(cols, name)), name)
end

checktabletimes(v::AbstractVector) =
    eltype(v) <: AbstractString ? throw(ArgumentError(textualtime("table", "table"))) :
    v

# Function barrier, typed on the time vector and the context's time type: the
# order check (within the partition, and against the previous partition's last
# time), the optional stable sort, and the window search. Returns the rows to
# keep — a range, or a slice of the sort permutation — whether a time past the
# window was seen, and the last time for the next partition.
function tablerows(times::AbstractVector, checkorder::Bool, sort::Bool,
    closed::Bool, prevtime, start, stop)
    if sort && !issorted(times)
        perm = stableperm(times)
        lo, hi = windowbounds(view(times, perm), closed, start, stop)
        # a sorted read is a single partition, so there is nothing to stop early
        return (perm[lo:hi], false, prevtime)
    end
    if checkorder && !sort
        issorted(times) ||
            throw(ArgumentError("time column in table is not non-decreasing"))
        if !isempty(times)
            prevtime === nothing || prevtime <= first(times) ||
                throw(
                    ArgumentError("time column in table is not non-decreasing \
                    across partitions"),
                )
            prevtime = last(times)
        end
    end
    lo, hi = windowbounds(times, closed, start, stop)
    return (lo:hi, hi < lastindex(times), prevtime)
end

# Ties keep their row order: rows sharing a timestamp form a cycle, and their
# order within it is data.
stableperm(times::AbstractVector) = sortperm(times; alg = Base.Sort.DEFAULT_STABLE)

# The window over sorted times: `[lo, hi]` indexes the rows in `[start, stop)`,
# or in `[start, stop]` when closed (empty when `hi < lo`).
function windowbounds(times::AbstractVector, closed::Bool, start, stop)
    lo = searchsortedfirst(times, start)
    hi = closed ? searchsortedlast(times, stop) : searchsortedfirst(times, stop) - 1
    return (lo, hi)
end

# The kept times, converted to the context's time type in the same single copy.
copytimes(::Type{T}, times::AbstractVector, rows) where {T} =
    copyto!(Vector{T}(undef, length(rows)), view(times, rows))

# The chunk, in the table's own column order: the time column where its source
# column stood (appended, for a function with no `:time` to overwrite), every
# other column indexed by `rows` — one dispatch and one copy per column.
function assemblechunk(cols, colnames, source, timecol::Vector, rows)
    timename = something(source, :time)
    outnames = Symbol[]
    outcols = AbstractVector[]
    for n in colnames
        push!(outnames, n === timename ? :time : n)
        push!(outcols, n === timename ? timecol : Tables.getcolumn(cols, n)[rows])
    end
    if source === nothing && !(:time in colnames)
        push!(outnames, :time)
        push!(outcols, timecol)
    end
    return DataFrame(outcols, outnames; copycols = false)
end

# The run function over time-ordered, non-empty chunks, shared by the frame and
# the DataFrame paths. `fctx` is the frame's context (nothing for a DataFrame,
# which has none to check against); `closed = nothing` closes the window when
# the run's stop is the frame's. `share` lets a chunk wholly inside the window
# go on as a private index over its vectors; without it every chunk is sliced.
function framepipeline(chunks::Vector{DataFrame}, fctx::Union{Nothing,Context},
    closed::Union{Nothing,Bool}, checkcontext::Bool, share::Bool)
    return CausalPipeline() do ctx::Context
        checkcontext && checkframecontext(ctx, fctx)
        isclosed = closed === nothing ? ctx.stop == fctx.stop : closed
        start, stop = ctx.start, ctx.stop
        i0 = firstchunk(c -> last(c.time) >= start, chunks)
        i1 =
            firstchunk(c -> isclosed ? first(c.time) > stop : first(c.time) >= stop,
                chunks) - 1
        return chunkmap(c -> clipframechunk(c, start, stop, isclosed, share),
            view(chunks, i0:i1))
    end
end

# Outside its context a frame's rows are unknown rather than absent, so reading
# there would pass off missing data as an empty stretch.
function checkframecontext(ctx::Context, fctx::Context)
    ctx.start < fctx.start || ctx.stop > fctx.stop || return nothing
    throw(
        ArgumentError("readtable: context [$(ctx.start), $(ctx.stop)) is not \
            within the frame's context [$(fctx.start), $(fctx.stop)], outside \
            which its rows are unknown; pass checkcontext = false to clip to it \
            anyway"),
    )
end

# Binary search over time-ordered chunks: the first index at which `pred`, false
# then true along the chunks, holds — or one past the end.
function firstchunk(pred, chunks::Vector{DataFrame})
    lo, hi = 1, length(chunks) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if pred(chunks[mid])
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end

# Consumers may mutate a chunk's column index but never a column vector (see
# DESIGN.md, "CSV output"), so a shared chunk goes on as a private index over
# the same vectors — O(ncols), no row copied.
function clipframechunk(c::DataFrame, start::T, stop::T, closed::Bool,
    share::Bool) where {T}
    lo, hi = windowbounds(c.time, closed, start, stop)
    hi < lo && return nothing
    out = share && lo == 1 && hi == nrow(c) ? DataFrame(c; copycols = false) :
          c[lo:hi, :]
    eltype(out.time) <: T || (out[!, :time] = convert(Vector{T}, out.time))
    return out
end
