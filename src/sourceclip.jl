# The clip shared by every file source (readcsv, readparquet, readjls):
# rename columns, resolve `:time`, check order, and keep the rows in the
# window.

# The configuration and running state shared by every file source, and the
# argument `clipchunk!` and `gatherchunk!` take: where rows come from, how
# `:time` is resolved, which rows are kept, and, across chunks, the last raw
# time seen and whether the stream is done. A file-source keyword that bears on
# the clip is a field here. The untyped fields are read once per chunk; per-row
# work sits behind `clipchunk!`'s function barriers.
mutable struct SourceClip{T}
    const path::String
    const what::String  # the format, for messages: "CSV file", "parquet file", …
    const time::Any     # Nothing | Symbol (column name) | Function (row -> time)
    const rename::Any   # Nothing | AbstractDict/map | Function (name -> name)
    const closed::Bool
    const skipmissing::Bool
    const columns::Union{Nothing,Vector{Symbol}}  # kept besides :time; nothing: all
    const start::T
    const stop::T
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool          # a time past the window was seen, or the input ended
end

SourceClip(ctx::Context{T}, path::AbstractString, what::String, time, rename,
    closed::Bool, skipmissing::Bool, columns = nothing) where {T} =
    SourceClip{T}(String(path), what, time, rename, closed, skipmissing,
        columns, ctx.start, ctx.stop, nothing, false)

# A file source's `columns` keyword, checked eagerly: `nothing` for every
# column, else the names to keep (after `rename`), as a fresh vector. A listed
# `:time` is the output's, always kept, so it is dropped here rather than read
# as a file column clashing with the one `time` names.
function sourcecolumns(columns, op::String)
    columns === nothing && return nothing
    columns isa Union{AbstractVector{Symbol},Tuple{Vararg{Symbol}}} ||
        throw(ArgumentError("invalid $op columns of type $(typeof(columns)): \
            expected a vector or tuple of Symbols, or `nothing`"))
    names = collect(Symbol, columns)
    allunique(names) || throw(ArgumentError("$op columns must be unique"))
    return filter!(!=(:time), names)
end

# Whether a source reads the file column `name`: every column without
# `columns`, else one that `rename` makes a listed name or the time column
# (`time` when a Symbol, else `:time`). The readers push this down, so unread
# columns are never decoded; `keepcolumns!` holds every chunk to it regardless.
readscolumn(clip::SourceClip, name::String) =
    clip.columns === nothing ||
    keepscolumn(clip, Symbol(renamedto(name, clip.rename)))
readcolumns(clip::SourceClip, filenames::Vector{String}) =
    filter(n -> readscolumn(clip, n), filenames)
keepscolumn(clip::SourceClip, name::Symbol) =
    name in clip.columns || name === timetarget(clip.time)
timetarget(time::Symbol) = time
timetarget(time) = :time

# Drop the columns `columns` leaves out, once renamed, refusing a listed one
# the file lacks.
function keepcolumns!(df::DataFrame, clip::SourceClip)
    names = clip.columns
    names === nothing && return nothing
    for name in names
        hascolumn(df, name) || throw(
            ArgumentError(
                "$(sourcename(clip.what, clip.path)) has no column $(repr(name))"),
        )
    end
    keep = filter(n -> keepscolumn(clip, n), propertynames(df))
    length(keep) == ncol(df) || select!(df, keep)
    return nothing
end

# The first steps of every chunk, sorted or not: rename the columns, keep
# `columns`, materialize `:time`, and find the rows whose time is present
# (`nothing` for all), refusing a missing one unless `skipmissing`.
function resolvechunk!(clip::SourceClip, df::DataFrame)
    renamecolumns!(df, clip.rename)
    keepcolumns!(df, clip)
    resolvetime!(df, clip.time, clip.path, clip.what)
    return presentrows(df.time, clip.skipmissing, clip.what, clip.path)
end

# Shared by the file sources: rename columns, materialize `:time`, drop (or
# refuse) missing-time rows, check order within the chunk and against the
# previous one, clip to the window, and convert `:time` to the context's time
# type. Returns nothing when no row is in the window. A time past the window
# marks the clip done, which ends the source.
function clipchunk!(clip::SourceClip{T}, df::DataFrame) where {T}
    what, path = clip.what, clip.path
    # Missing rows are excluded through the row index rather than deleted
    # first, so the chunk is copied once.
    present = resolvechunk!(clip, df)
    times = present === nothing ? df.time : view(df.time, present)
    issorted(times) || throw(ArgumentError(unordered(what, path)))
    if !isempty(times)
        clip.prevtime !== nothing && first(times) < clip.prevtime &&
            throw(ArgumentError(unordered(what, path)))
        clip.prevtime = last(times)
    end
    lo, hi = windowbounds(times, clip.closed, clip.start, clip.stop)
    hi < length(times) && (clip.done = true)   # saw a time past the window
    hi < lo && return nothing
    # The chunk is freshly read and owned, so keeping every row needs no copy.
    clipped =
        present !== nothing ? df[present[lo:hi], :] :
        lo == 1 && hi == nrow(df) ? df : df[lo:hi, :]
    clipped[!, :time] = convert(Vector{T}, clipped.time)
    return clipped
end

# The next non-empty clipped chunk pulled from `chunks`, a cursor over the
# source's freshly read DataFrames; the clip is done once they run out.
function nextclipped!(clip::SourceClip, chunks::PullCursor)
    while !clip.done
        df = pull!(chunks)
        df === nothing && break
        out = clipchunk!(clip, df)
        out === nothing || return out
    end
    clip.done = true
    return nothing
end

# A file producer's next chunk from its cursor over freshly read DataFrames:
# streamed through the clip, or, when the read must `gather` (`sort = true`
# without sorting at the source), every in-window row sorted into one chunk.
# `nothing` once the clip is done.
function pullclipped!(clip::SourceClip, chunks::PullCursor, gather::Bool)
    clip.done && return nothing
    return gather ? gathersorted!(clip, chunks) : nextclipped!(clip, chunks)
end

# A `sort = true` read, which cannot stream: every chunk `chunks` yields is
# scanned for in-window rows, which are sorted once into a single chunk.
function gathersorted!(clip::SourceClip{T}, chunks::PullCursor) where {T}
    clip.done = true
    kept = DataFrame[]
    while (df = pull!(chunks)) !== nothing
        gatherchunk!(kept, clip, df)
    end
    return sortgathered(kept, T)
end

# `clipchunk!`'s counterpart for a source asked to `sort`: the same rename and
# time resolution, but with no order check, binary search or early stop. The
# in-window rows, found by a scan, are pushed onto `kept` to be sorted once the
# file is exhausted.
function gatherchunk!(kept::Vector{DataFrame}, clip::SourceClip, df::DataFrame)
    present = resolvechunk!(clip, df)
    closed, start, stop = clip.closed, clip.start, clip.stop
    rows =
        present === nothing ? windowrows(df.time, closed, start, stop) :
        present[windowrows(view(df.time, present), closed, start, stop)]
    if length(rows) == nrow(df)
        push!(kept, df)    # owned, as in clipchunk!
    elseif !isempty(rows)
        push!(kept, df[rows, :])
    end
    return nothing
end

# Function barrier, typed on the raw time vector: the indices of the rows in
# `[start, stop)`, or `[start, stop]` when closed, in file order.
windowrows(times::AbstractVector, closed::Bool, start, stop) =
    closed ? findall(t -> start <= t && t <= stop, times) :
    findall(t -> start <= t && t < stop, times)

# The gathered rows as one chunk stably sorted by time, so ties keep file
# order, with `:time` converted to the context's time type; nothing if empty.
function sortgathered(kept::Vector{DataFrame}, ::Type{T}) where {T}
    isempty(kept) && return nothing
    df = vcatowned(kept)
    issorted(df.time) || (df = df[stableperm(df.time), :])
    df[!, :time] = convert(Vector{T}, df.time)
    return df
end

# Owned pieces as one owned DataFrame. A lone piece is returned as is, where
# `reduce(vcat, …)` would copy it.
vcatowned(pieces::Vector{DataFrame}) =
    length(pieces) == 1 ? only(pieces) : reduce(vcat, pieces)

# The subject of every data error a source raises: "CSV file x.csv", or a bare
# "table" for readtable, which has no path. Built only on the error path.
sourcename(what::String, path::String) = isempty(path) ? what : "$what $path"

unordered(
    what::String,
    path::String,
) = "time column in $(sourcename(what, path)) is not non-decreasing"

# The error for a textual time column, which can't be ordered against the
# window. The hint depends on the cause: a `time` function to fix, CSV's
# `types`, or, for sources carrying their own types, a `time` function.
function textualtime(what::String, path::String, fromfunction::Bool)
    src = sourcename(what, path)
    fromfunction &&
        return "the `time` function over $src returned text; return a value \
            ordered like the context's times"
    what == "CSV file" &&
        return "time column in $src needs a concrete type via `types`, or a \
            `time` function to produce it"
    return "time column in $src is textual; use a `time` function to produce \
        a usable time"
end

# Whether the non-missing eltype is text. An all-missing column (nonmissingtype
# `Union{}`) is not.
function istextual(v::AbstractVector)
    S = nonmissingtype(eltype(v))
    return S !== Union{} && S <: AbstractString
end

# Rename columns before the time column is resolved. A map renames only the
# columns it names; a function is applied to every column name.
renamecolumns!(::DataFrame, ::Nothing) = nothing
renamecolumns!(df::DataFrame, rename) = (rename!(n -> renamedto(n, rename), df); nothing)

# A column's name after `rename`. A map may be keyed by String or Symbol names,
# so both are looked up.
renamedto(n::String, ::Nothing) = n
renamedto(n::String, f) = String(f(n))
function renamedto(n::String, m::AbstractDict)
    haskey(m, n) && return String(m[n])
    haskey(m, Symbol(n)) && return String(m[Symbol(n)])
    return n
end

# Materialize `:time`, from a column or a function, and reject it if textual.
function resolvetime!(df::DataFrame, time, path::String, what::String)
    if time isa Function
        df[!, :time] =
            checktimes(maptime(time, Tables.columntable(df)), true, what, path)
    else
        name = timecolumn(df, time, what, path)
        name === :time || rename!(df, name => :time)
        checktimes(df.time, false, what, path)
    end
    return nothing
end

# The raw time values, and the column they came from (`nothing` for a `time`
# function, whose values become a `:time` column), refused if textual. `cols`
# is a DataFrame or a Tables.jl column table.
sourcetimes(cols, time::Function, what::String, path::String) =
    (checktimes(maptime(time, Tables.columntable(cols)), true, what, path), nothing)
function sourcetimes(cols, time::Union{Nothing,Symbol}, what::String, path::String)
    name = timecolumn(cols, time, what, path)
    return (checktimes(Tables.getcolumn(cols, name), false, what, path), name)
end

# The column holding the times, which must exist and, unless it is `:time`
# itself, must not clash with a `:time` column.
function timecolumn(cols, time::Union{Nothing,Symbol}, what::String, path::String)
    name = something(time, :time)
    hascolumn(cols, name) || throw(
        ArgumentError(
            time === nothing ?
            "$(sourcename(what, path)) has no time column; choose one with `time`" :
            "$(sourcename(what, path)) has no column $(repr(name))"),
    )
    name === :time || !hascolumn(cols, :time) ||
        throw(ArgumentError(timeclash(what, path, name)))
    return name
end

# A DataFrame's `columnindex`, unlike its names, allocates nothing.
hascolumn(df::AbstractDataFrame, name::Symbol) = columnindex(df, name) > 0
hascolumn(cols, name::Symbol) = name in Tables.columnnames(cols)

checktimes(v::AbstractVector, fromfunction::Bool, what::String, path::String) =
    istextual(v) ? throw(ArgumentError(textualtime(what, path, fromfunction))) : v

# Make `times`, as `sourcetimes` found them, the frame's `:time` column.
function applytime!(df::DataFrame, times::AbstractVector, source)
    if source === nothing
        df[!, :time] = times
    elseif source !== :time
        rename!(df, source => :time)
    end
    return nothing
end

timeclash(
    what::String,
    path::String,
    name::Symbol,
) = "$(sourcename(what, path)) has both a :time column and the time column \
    $(repr(name)); drop one of them first"

# Function barrier: computes the time column by applying `f` to the concretely
# typed rows of the column table, so `f` specializes and the eltype is inferred.
maptime(f, nt::NamedTuple) = map(f, Tables.rows(nt))
