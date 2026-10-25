# `readtable`, the source over in-memory data. Three paths share one clip rule
# (`windowbounds`):
#
# - a generic Tables.jl table is resolved per run and per partition by
#   `TableProducer`, copying only the in-window rows;
# - an AbstractDataFrame is resolved once, at construction, into a private
#   index over the caller's vectors, then read by the frame path;
# - a CausalFrame's chunks are clipped as they stand (`framepipeline`), a chunk
#   wholly inside the window passed on as a private index over its vectors.

"""
    readtable(table; columns = nothing, time = nothing, checkorder = true,
              sort = false, closed = false, skipmissing = false)
        -> CausalPipeline
    readtable(frame::CausalFrame; closed = nothing, checkcontext = true)
        -> CausalPipeline

A source reading an in-memory table, clipped to `[start, stop)`. Only the rows
in the window are copied.

# Arguments
- `table`: any Tables.jl table — a `DataFrame`, a `NamedTuple` of vectors, a
  vector of `NamedTuple`s. A table with several `Tables.partitions` is read one
  chunk per partition. A `DataFrame` is referenced, not copied, and is prepared
  when `readtable` is called, so its time errors are raised there; do not mutate
  it while the pipeline is in use.
- `frame`: a loaded [`CausalFrame`](@ref), read back chunk by chunk. Its time is
  already resolved, so only `closed` and `checkcontext` apply.

# Keywords
- `columns = nothing`: the columns to read, as for [`readcsv`](@ref); the
  others are never copied.
- `time = nothing`: where `:time` comes from, as for `readcsv`. A textual time
  column is an `ArgumentError`.
- `checkorder = true`: check that time is non-decreasing. With `false` the
  caller vouches for the order, and an unsorted table gives wrong results rather
  than an error.
- `sort = false`: stably sort the rows by time first. A partitioned table is
  concatenated to sort it.
- `closed = false`: clip to `[start, stop]` instead, keeping rows at `stop`. For
  a frame the default `nothing` keeps them exactly when the run's `stop` equals
  the frame's own, so `load(context(frame), readtable(frame))` reproduces
  `frame`.
- `skipmissing = false`: drop rows whose time is `missing`. Without it such a
  row is an `ArgumentError`.
- `checkcontext = true`: for a frame, require the run's context to lie within
  `context(frame)` (an `ArgumentError` otherwise); `false` clips the frame to any
  context.

```jldoctest
df = DataFrame(ts = [3, 1, 2], bid = [1.0, 2.0, 3.0])
p = readtable(df; time = :ts, sort = true) |> filterrows(r -> r.bid > 1)
DataFrame(load(Context(0, 10), p))

# output

2×2 DataFrame
 Row │ time   bid
     │ Int64  Float64
─────┼────────────────
   1 │     1      2.0
   2 │     2      3.0
```
"""
function readtable(table; columns = nothing, time = nothing,
    checkorder::Bool = true, sort::Bool = false, closed::Bool = false,
    skipmissing::Bool = false)
    checksourcetimespec(time, "readtable")
    names = sourcecolumns(columns, "readtable")
    Tables.istable(table) || throw(ArgumentError("readtable table must be a \
        Tables.jl table, got $(typeof(table))"))
    return pipesource(; stopfree = !closed) do ctx::Context
        return ChunkSource(
            TableProducer{timetype(ctx)}(table, names, time, checkorder,
                sort, closed, skipmissing, ctx.start, ctx.stop),
        )
    end
end

function readtable(df::AbstractDataFrame; columns = nothing, time = nothing,
    checkorder::Bool = true, sort::Bool = false, closed::Bool = false,
    skipmissing::Bool = false)
    checksourcetimespec(time, "readtable")
    names = sourcecolumns(columns, "readtable")
    # A private index over the caller's vectors, so the projection, rename and
    # :time assignment below never reach `df`. O(ncols).
    wrapped = DataFrame(df; copycols = false)
    n = nrow(wrapped)
    keepcolumns!(wrapped, names, time, "table", "")
    times, source = sourcetimes(wrapped, time, "table", "", n)
    applytime!(wrapped, times, source)
    # Unless dropping missing times or sorting copies the rows, the chunk
    # aliases the caller's vectors, so every run must slice rather than share.
    owned = false
    present = presentrows(times, skipmissing, "table", "")
    if present !== nothing
        wrapped = wrapped[present, :]
        times = wrapped.time
        owned = true
    end
    if sort
        if !issorted(times)
            wrapped = wrapped[stableperm(times), :]
            owned = true
        end
    elseif checkorder
        issorted(times) ||
            throw(ArgumentError(unordered("table", "")))
    end
    chunks = nrow(wrapped) > 0 ? [wrapped] : DataFrame[]
    return framepipeline(chunks, nothing, closed, false, owned)
end

readtable(frame::CausalFrame; closed::Union{Nothing,Bool} = nothing,
    checkcontext::Bool = true) =
    framepipeline(frame.chunks, frame.context, closed, checkcontext, true)

# The generic path's producer. The untyped fields are per-partition setup; the
# per-row work sits behind `tablerows`, `copytimes` and `maptime`.
mutable struct TableProducer{T,X}
    const table::X
    const columns::Union{Nothing,Vector{Symbol}}  # kept besides :time; nothing: all
    const time::Any      # Nothing | Symbol (column name) | Function (row -> time)
    const checkorder::Bool
    const sort::Bool
    const closed::Bool
    const skipmissing::Bool
    const start::T
    const stop::T
    parts::Any           # PullCursor over the partitions, created on first pull
    prevtime::Any        # last time of the previous partition, for its order
    done::Bool
    TableProducer{T}(table::X, columns, time, checkorder, sort, closed,
        skipmissing, start, stop) where {T,X} =
        new{T,X}(table, columns, time, checkorder, sort, closed, skipmissing,
            start, stop, nothing, nothing, false)
end

function (p::TableProducer)()
    p.parts === nothing && (p.parts = PullCursor(tablepartitions(p)))
    while !p.done
        part = pull!(p.parts)
        part === nothing && (p.done = true; break)
        chunk = tablechunk!(p, part)
        chunk === nothing || return chunk
    end
    return nothing
end

# The partitions are the chunks, except that a sort concatenates them first,
# each narrowed to the columns `columns` keeps so the others aren't copied.
function tablepartitions(p::TableProducer)
    parts = Tables.partitions(p.table)
    p.sort || return parts
    collected = collect(parts)
    length(collected) <= 1 && return collected
    return (reduce(vcat, [keptpartition(p, x) for x in collected]),)
end

# A partition as an uncopied DataFrame of the columns `columns` keeps or, with
# none kept, its first, to carry the row count as `keptfilecolumns` does.
function keptpartition(p::TableProducer, part)
    df = DataFrame(part; copycols = false)
    p.columns === nothing && return df
    keep = keptcolumns(p.columns, p.time, propertynames(df), "table", "")
    return select(df, isempty(keep) ? first(propertynames(df), 1) : keep;
        copycols = false)
end

# One partition's in-window rows as a chunk, or nothing if none. Carries the
# last time to the next partition's order check, and ends the source on a time
# past the window.
function tablechunk!(p::TableProducer{T}, part) where {T}
    cols = Tables.columns(part)
    colnames = Tables.columnnames(cols)
    n = isempty(colnames) ? 0 : length(Tables.getcolumn(cols, first(colnames)))
    if p.columns !== nothing
        colnames = Tuple(keptcolumns(p.columns, p.time, colnames, "table", ""))
        cols = NamedTuple{colnames}(map(c -> Tables.getcolumn(cols, c), colnames))
    end
    times, source = sourcetimes(cols, p.time, "table", "", n)
    present = presentrows(times, p.skipmissing, "table", "")
    # with missing times dropped, the kept indices are into the view of the
    # present rows, mapped back to the partition below
    kept, sawstop, p.prevtime =
        tablerows(present === nothing ? times : view(times, present),
            p.checkorder, p.sort, p.closed, p.prevtime, p.start, p.stop)
    sawstop && (p.done = true)
    rows = present === nothing ? kept : present[kept]
    isempty(rows) && return nothing
    return assemblechunk(cols, colnames, source, copytimes(T, times, rows), rows)
end

# Function barrier over the time vector: the order check (within the partition
# and against the previous one), the optional stable sort, and the window
# search. Returns the rows to keep (a range or a slice of the sort permutation),
# whether a time past the window was seen, and the last time.
function tablerows(times::AbstractVector, checkorder::Bool, sort::Bool,
    closed::Bool, prevtime, start, stop)
    if sort && !issorted(times)
        perm = stableperm(times)
        lo, hi = windowbounds(view(times, perm), closed, start, stop)
        # a sorted read is a single partition, so there is nothing to stop early
        return (perm[lo:hi], false, prevtime)
    end
    if checkorder && !sort
        issorted(times) || throw(ArgumentError(unordered("table", "")))
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

# Function barrier over the raw times, shared by the sources: `nothing` when no
# time is missing, else the indices of the present ones under `skipmissing`,
# and an error without it. `what` and `path` name the source in the error.
function presentrows(times::AbstractVector, skipmissing::Bool, what::String,
    path::String)
    Missing <: eltype(times) || return nothing
    any(ismissing, times) || return nothing
    skipmissing || throw(
        ArgumentError("time column in $(sourcename(what, path)) has missing \
            values; pass `skipmissing = true` to drop those rows"),
    )
    return findall(.!ismissing.(times))   # a BitVector, so findall sizes once
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

# The kept times, copied and converted to the context's time type in one pass.
copytimes(::Type{T}, times::AbstractVector, rows) where {T} =
    copyto!(Vector{T}(undef, length(rows)), view(times, rows))

# The chunk, in the table's column order: `timecol` where its source column
# stood (appended when a function produced it and there is no `:time`), every
# other column indexed by `rows`.
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

# The pipeline over time-ordered, non-empty chunks, for the frame and DataFrame
# paths. `fctx` is the frame's context (nothing for a DataFrame);
# `closed = nothing` closes the window when the run's stop is the frame's.
# `share` lets a chunk wholly inside the window go on as a private index over
# its vectors; without it every chunk is sliced.
function framepipeline(chunks::Vector{DataFrame}, fctx::Union{Nothing,Context},
    closed::Union{Nothing,Bool}, checkcontext::Bool, share::Bool)
    # `closed = nothing` closes only a run stopping where the frame does, and a
    # reader stopping earlier is cut at its own stop, so it keeps stopfree.
    return pipesource(; stopfree = closed !== true) do ctx::Context
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
            which its rows are unknown; pass `checkcontext = false` to clip to it \
            anyway"),
    )
end

# Binary search for the first chunk satisfying `pred`, which must be monotone
# (false, then true); one past the end if none.
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

# Consumers may mutate a chunk's column index but never a column vector (the
# `writecsv` hand-off argument), so a shared chunk goes on as a private index
# over the same vectors.
function clipframechunk(c::DataFrame, start::T, stop::T, closed::Bool,
    share::Bool) where {T}
    lo, hi = windowbounds(c.time, closed, start, stop)
    hi < lo && return nothing
    out = share && lo == 1 && hi == nrow(c) ? DataFrame(c; copycols = false) :
          c[lo:hi, :]
    eltype(out.time) <: T || (out[!, :time] = convert(Vector{T}, out.time))
    return out
end
