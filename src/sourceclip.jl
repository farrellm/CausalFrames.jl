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
    const start::T
    const stop::T
    prevtime::Any       # last raw time seen, for cross-chunk sortedness
    done::Bool          # a time past the window was seen, or the input ended
end

SourceClip(ctx::Context{T}, path::AbstractString, what::String, time, rename,
    closed::Bool, skipmissing::Bool) where {T} =
    SourceClip{T}(String(path), what, time, rename, closed, skipmissing,
        ctx.start, ctx.stop, nothing, false)

# Shared by the file sources: rename columns, materialize `:time`, drop (or
# refuse) missing-time rows, check order within the chunk and against the
# previous one, clip to the window, and convert `:time` to the context's time
# type. Returns nothing when no row is in the window. A time past the window
# marks the clip done, which ends the source.
function clipchunk!(clip::SourceClip{T}, df::DataFrame) where {T}
    what, path = clip.what, clip.path
    renamecolumns!(df, clip.rename)
    resolvetime!(df, clip.time, path, what)
    # Missing rows are excluded through the row index rather than deleted
    # first, so the chunk is copied once.
    present = presentrows(df.time, clip.skipmissing, what, path)
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

# `clipchunk!`'s counterpart for a source asked to `sort`: the same rename and
# time resolution, but with no order check, binary search or early stop. The
# in-window rows, found by a scan, are pushed onto `kept` to be sorted once the
# file is exhausted.
function gatherchunk!(kept::Vector{DataFrame}, clip::SourceClip, df::DataFrame)
    what, path = clip.what, clip.path
    renamecolumns!(df, clip.rename)
    resolvetime!(df, clip.time, path, what)
    present = presentrows(df.time, clip.skipmissing, what, path)
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
    df = length(kept) == 1 ? only(kept) : reduce(vcat, kept)
    issorted(df.time) || (df = df[stableperm(df.time), :])
    df[!, :time] = convert(Vector{T}, df.time)
    return df
end

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
renamecolumns!(df::DataFrame, f) = (rename!(f, df); nothing)
# A map may be keyed by String or Symbol names, so both are looked up.
function renamecolumns!(df::DataFrame, m::AbstractDict)
    pairs = Pair{String,Symbol}[]
    for n in names(df)
        s = Symbol(n)
        haskey(m, n) ? push!(pairs, n => Symbol(m[n])) :
        haskey(m, s) && push!(pairs, n => Symbol(m[s]))
    end
    isempty(pairs) || rename!(df, pairs)
    return nothing
end

# Materialize `:time`, from a column or a function, and reject it if textual.
# `columnindex`, unlike `names(df)`, allocates nothing.
function resolvetime!(df::DataFrame, time, path::String, what::String)
    if time isa Function
        df[!, :time] = maptime(time, Tables.columntable(df))
    else
        if time isa Symbol && time !== :time
            columnindex(df, time) > 0 ||
                throw(ArgumentError("$what $path has no column $(repr(time))"))
            columnindex(df, :time) > 0 &&
                throw(ArgumentError(timeclash(what, path, time)))
            rename!(df, time => :time)
        end
        columnindex(df, :time) > 0 || throw(
            ArgumentError("$what $path has no time column; choose one with `time`"))
    end
    istextual(df.time) &&
        throw(ArgumentError(textualtime(what, path, time isa Function)))
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
