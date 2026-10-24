# Min/Max/First/Last each track one value of the input column's type, with no
# identity element, so they share a state. `F` is the singleton type of the
# combiner (min, max, keepfirst, keeplast), recovered as `F.instance`, so
# `update!` specializes per summarizer. `val` is typed exactly as the input
# column and left *undefined* until the first row; `seen` guards every read.
# `value` is never called before a row is folded: the transforms answer the
# no-rows case with `emptyvalue`.

keepfirst(a, b) = a
keeplast(a, b) = b

mutable struct TrackState{C,N,T,F} <: SummarizerState
    seen::Bool
    val::T
    TrackState{C,N,T,F}() where {C,N,T,F} = new{C,N,T,F}(false)
    TrackState{C,N,T,F}(seen::Bool, val) where {C,N,T,F} = new{C,N,T,F}(seen, val)
end

fresh(::S) where {S<:TrackState} = S()
# Only `seen` is cleared; the stale `val` is never read.
@inline fresh!(st::TrackState) = (st.seen = false; st)
@inline function update!(st::TrackState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    st.val = st.seen ? F.instance(st.val, v) : v
    st.seen = true
    return nothing
end
# Reads both inputs before writing, so `dest` may alias `a` or `b`. `a`'s rows
# precede `b`'s, which makes keepfirst/keeplast correct.
function combine!(dest::TrackState{C,N,T,F}, a::TrackState{C,N,T,F},
    b::TrackState{C,N,T,F}) where {C,N,T,F}
    if a.seen || b.seen
        dest.val = !b.seen ? a.val : !a.seen ? b.val : F.instance(a.val, b.val)
    end
    dest.seen = a.seen || b.seen
    return nothing
end
value(st::TrackState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))
function widenstate(st::TrackState{C,N,T,F}, intypes::NamedTuple) where {C,N,T,F}
    T2 = intypes[C]
    T2 === T && return st
    return st.seen ? TrackState{C,N,T2,F}(true, convert(T2, st.val)) :
           TrackState{C,N,T2,F}()
end

# The windowed state of Min, Max and First: a monotonic deque of the window's
# candidate values, oldest at the front. A new value `v` first pops every value
# `b` at the back with `F(b, v)` equal to `v`, which can't be the answer while
# `v` is in the window, so the front is always the window's fold. This needs
# `F` to select one of its arguments associatively, as `min` and `max` do under
# `isequal` (NaN, ±0.0 and `missing` included), and `keepfirst` does (nothing
# is popped, so the deque is the window). Last has a cheaper state below.
#
# Rows leave oldest first, so eviction needs only sequence numbers: the front
# goes when its row does, and a popped row is already gone. Dead front slots
# are reclaimed once they dominate, so a steady window doesn't allocate.
#
# `K` is what `value` reports: `:value` the front's value (Min, Max, First),
# `:index` its rows-since, `pushed - seqs[front]` (MinIndex, MaxIndex), or
# `:both` (MinWithIndex, MaxWithIndex), when `N` is a pair of names. A tie pops
# the older value, so the most recent extreme is the one reported.
mutable struct WindowTrackState{C,N,T,F,K} <: SummarizerState
    vals::Vector{T}
    seqs::Vector{Int}  # each live value's row number, counted from the last zero
    front::Int         # first live slot
    pushed::Int        # rows folded since the last zero
    popped::Int        # rows removed since the last zero
end

WindowTrackState{C,N,T,F,K}() where {C,N,T,F,K} =
    WindowTrackState{C,N,T,F,K}(T[], Int[], 1, 0, 0)

fresh(::S) where {S<:WindowTrackState} = S()
@inline function fresh!(st::WindowTrackState)
    empty!(st.vals)
    empty!(st.seqs)
    st.front = 1
    st.pushed = 0
    st.popped = 0
    return st
end
@inline function update!(st::WindowTrackState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    vals, seqs = st.vals, st.seqs
    while length(vals) >= st.front && isequal(F.instance(@inbounds(vals[end]), v), v)
        pop!(vals)
        pop!(seqs)
    end
    push!(vals, v)
    push!(seqs, st.pushed += 1)
    return nothing
end
@inline function downdate!(st::WindowTrackState, row)
    st.popped += 1
    seqs = st.seqs
    if st.front <= length(seqs) && @inbounds(seqs[st.front]) == st.popped
        st.front += 1
        dead = compactdead!(seqs, st.front - 1, 32)
        if dead > 0
            dropfront!(st.vals, dead)
            st.front = 1
        end
    end
    return nothing
end
value(st::WindowTrackState{C,N,T,F,:value}) where {C,N,T,F} =
    NamedTuple{(N,),Tuple{T}}((@inbounds(st.vals[st.front]),))
value(st::WindowTrackState{C,N,T,F,:index}) where {C,N,T,F} =
    NamedTuple{(N,),Tuple{Int}}((st.pushed - @inbounds(st.seqs[st.front]),))
value(st::WindowTrackState{C,N,T,F,:both}) where {C,N,T,F} =
    NamedTuple{N,Tuple{T,Int}}((@inbounds(st.vals[st.front]),
        st.pushed - @inbounds(st.seqs[st.front])))

"""
    Min(column::ColumnSpec) -> Summarizer

The minimum of `column`, in `:{column}_min`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the minimum, up to the whole window when `column` rises.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Min{C} <: GroupSummarizer end
Min(column::ColumnSpec) = withterms(Min{colname(column)}(), column)

emptyvalue(::Min{C}) where {C} = NamedTuple{(Symbol(C, :_min),)}((missing,))
fresh(::Min{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_min),intypes[C],typeof(min)}()

"""
    Max(column::ColumnSpec) -> Summarizer

The maximum of `column`, in `:{column}_max`, with `column`'s element type
(`missing` for no rows). In a sliding window it keeps the values that could
still become the maximum, up to the whole window when `column` falls.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Max{C} <: GroupSummarizer end
Max(column::ColumnSpec) = withterms(Max{colname(column)}(), column)

emptyvalue(::Max{C}) where {C} = NamedTuple{(Symbol(C, :_max),)}((missing,))
fresh(::Max{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_max),intypes[C],typeof(max)}()

"""
    First(column::ColumnSpec) -> Summarizer

The value of `column` in the first row folded, in `:{column}_first`, with
`column`'s element type (`missing` for no rows). In a sliding window it keeps
every value in the window, O(window) memory per key.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct First{C} <: GroupSummarizer end
First(column::ColumnSpec) = withterms(First{colname(column)}(), column)

emptyvalue(::First{C}) where {C} = NamedTuple{(Symbol(C, :_first),)}((missing,))
fresh(::First{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_first),intypes[C],typeof(keepfirst)}()

"""
    Last(column::ColumnSpec) -> Summarizer

The value of `column` in the last row folded, in `:{column}_last`, with
`column`'s element type (`missing` for no rows).

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).
"""
struct Last{C} <: GroupSummarizer end
Last(column::ColumnSpec) = withterms(Last{colname(column)}(), column)

emptyvalue(::Last{C}) where {C} = NamedTuple{(Symbol(C, :_last),)}((missing,))
fresh(::Last{C}, intypes::NamedTuple) where {C} =
    TrackState{C,Symbol(C, :_last),intypes[C],typeof(keeplast)}()

windowtrack(::TrackState{C,N,T,F}) where {C,N,T,F} =
    WindowTrackState{C,N,T,F,:value}()

# Last's windowed state: the newest value and a row count. Evicting the oldest
# row changes the last value only by emptying the window, so a count (not a
# time, which can't tell tied rows apart) is all eviction needs. `val` is
# undefined until the first row and stale once the window empties; `value` is
# only read with rows folded. About 3-4x cheaper per row than the deque.
mutable struct WindowLastState{C,N,T} <: SummarizerState
    n::Int
    val::T
    WindowLastState{C,N,T}() where {C,N,T} = new{C,N,T}(0)
end

freshwindowed(::Last{C}, intypes::NamedTuple) where {C} =
    WindowLastState{C,Symbol(C, :_last),intypes[C]}()
fresh(::S) where {S<:WindowLastState} = S()
@inline fresh!(st::WindowLastState) = (st.n = 0; st)
@inline function update!(st::WindowLastState{C}, row) where {C}
    st.val = getproperty(row, C)
    st.n += 1
    return nothing
end
@inline downdate!(st::WindowLastState, row) = (st.n -= 1; nothing)
value(st::WindowLastState{C,N,T}) where {C,N,T} = NamedTuple{(N,),Tuple{T}}((st.val,))

# MinIndex/MaxIndex and MinWithIndex/MaxWithIndex report where the extreme is,
# as rows since it (0 = the newest row): an absolute row number means nothing
# in a stream. Their ordinary state keeps the extreme, its rows-since `pos`
# and the count `n` of rows folded, which `combine!` shifts the older side's
# position by. `K` is `:index` or `:both`, as for WindowTrackState, which
# slides them. The new value wins when `F` selects it under `isequal`, so a
# tie goes to the most recent row, matching the deque, and `NaN` and `missing`
# win as they do for Min and Max. `val` is undefined while `n == 0`.
mutable struct IndexState{C,N,T,F,K} <: SummarizerState
    n::Int
    pos::Int
    val::T
    IndexState{C,N,T,F,K}() where {C,N,T,F,K} = new{C,N,T,F,K}(0, 0)
    IndexState{C,N,T,F,K}(n::Int, pos::Int, val) where {C,N,T,F,K} =
        new{C,N,T,F,K}(n, pos, val)
end

fresh(::S) where {S<:IndexState} = S()
# Only `n` is cleared; `update!` resets `pos` with the first row.
@inline fresh!(st::IndexState) = (st.n = 0; st)
@inline function update!(st::IndexState{C,N,T,F}, row) where {C,N,T,F}
    v = getproperty(row, C)
    if st.n == 0 || isequal(F.instance(st.val, v), v)
        st.val = v
        st.pos = 0
    else
        st.pos += 1
    end
    st.n += 1
    return nothing
end
# Reads both inputs before writing, so `dest` may alias `a` or `b`. `a`'s rows
# precede `b`'s: `a`'s extreme is `b.n` rows further back, and loses ties.
function combine!(dest::IndexState{C,N,T,F,K}, a::IndexState{C,N,T,F,K},
    b::IndexState{C,N,T,F,K}) where {C,N,T,F,K}
    an, bn = a.n, b.n
    if bn > 0 && (an == 0 || isequal(F.instance(a.val, b.val), b.val))
        dest.val = b.val
        dest.pos = b.pos
    elseif an > 0
        dest.val = a.val
        dest.pos = a.pos + bn
    end
    dest.n = an + bn
    return nothing
end
value(st::IndexState{C,N,T,F,:index}) where {C,N,T,F} =
    NamedTuple{(N,),Tuple{Int}}((st.pos,))
value(st::IndexState{C,N,T,F,:both}) where {C,N,T,F} =
    NamedTuple{N,Tuple{T,Int}}((st.val, st.pos))
function widenstate(st::IndexState{C,N,T,F,K}, intypes::NamedTuple) where {C,N,T,F,K}
    T2 = intypes[C]
    T2 === T && return st
    return st.n > 0 ? IndexState{C,N,T2,F,K}(st.n, st.pos, convert(T2, st.val)) :
           IndexState{C,N,T2,F,K}()
end
windowtrack(::IndexState{C,N,T,F,K}) where {C,N,T,F,K} =
    WindowTrackState{C,N,T,F,K}()

"""
    MaxIndex(column::ColumnSpec) -> Summarizer

How many rows ago `column` took its maximum, in `:{column}_maxindex`, as an
`Int` (`missing` for no rows): `0` is the newest row. Ties go to the most recent
row. A `NaN` or `missing` is the maximum, as for [`Max`](@ref), so its position
is reported while it is folded. To get the maximum as well, use
[`MaxWithIndex`](@ref), which keeps one state for both.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Examples
```jldoctest
df = DataFrame(time = 1:6, x = [3, 5, 2, 5, 1, 4])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MaxIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×3 DataFrame
 Row │ time   x      w3_x_maxindex
     │ Int64  Int64  Int64?
─────┼─────────────────────────────
   1 │     1      3              0
   2 │     2      5              0
   3 │     3      2              1
   4 │     4      5              0
   5 │     5      1              1
   6 │     6      4              2
```
"""
struct MaxIndex{C} <: GroupSummarizer end
MaxIndex(column::ColumnSpec) = withterms(MaxIndex{colname(column)}(), column)

emptyvalue(::MaxIndex{C}) where {C} =
    NamedTuple{(Symbol(C, :_maxindex),)}((missing,))
fresh(::MaxIndex{C}, intypes::NamedTuple) where {C} =
    IndexState{C,Symbol(C, :_maxindex),intypes[C],typeof(max),:index}()

"""
    MinIndex(column::ColumnSpec) -> Summarizer

How many rows ago `column` took its minimum, in `:{column}_minindex`, as an
`Int` (`missing` for no rows): `0` is the newest row. Ties go to the most recent
row. A `NaN` or `missing` is the minimum, as for [`Min`](@ref), so its position
is reported while it is folded. To get the minimum as well, use
[`MinWithIndex`](@ref), which keeps one state for both.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Examples
```jldoctest
df = DataFrame(time = 1:6, x = [3, 1, 4, 1, 5, 9])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MinIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×3 DataFrame
 Row │ time   x      w3_x_minindex
     │ Int64  Int64  Int64?
─────┼─────────────────────────────
   1 │     1      3              0
   2 │     2      1              0
   3 │     3      4              1
   4 │     4      1              0
   5 │     5      5              1
   6 │     6      9              2
```
"""
struct MinIndex{C} <: GroupSummarizer end
MinIndex(column::ColumnSpec) = withterms(MinIndex{colname(column)}(), column)

emptyvalue(::MinIndex{C}) where {C} =
    NamedTuple{(Symbol(C, :_minindex),)}((missing,))
fresh(::MinIndex{C}, intypes::NamedTuple) where {C} =
    IndexState{C,Symbol(C, :_minindex),intypes[C],typeof(min),:index}()

"""
    MaxWithIndex(column::ColumnSpec) -> Summarizer

The maximum of `column` and how many rows ago it was taken, in `:{column}_max`
(with `column`'s element type) and `:{column}_maxindex` (an `Int`), both
`missing` for no rows. It folds one state for both, where [`Max`](@ref) and
[`MaxIndex`](@ref) together fold two, and so can't be requested with `Max` of
the same column. The index follows `MaxIndex`.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Examples
```jldoctest
df = DataFrame(time = 1:6, x = [3, 5, 2, 5, 1, 4])
p = readtable(df) |> addrollingcolumns((w3 = 3,), MaxWithIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

6×4 DataFrame
 Row │ time   x      w3_x_max  w3_x_maxindex
     │ Int64  Int64  Int64?    Int64?
─────┼───────────────────────────────────────
   1 │     1      3         3              0
   2 │     2      5         5              0
   3 │     3      2         5              1
   4 │     4      5         5              0
   5 │     5      1         5              1
   6 │     6      4         5              2
```
"""
struct MaxWithIndex{C} <: GroupSummarizer end
MaxWithIndex(column::ColumnSpec) =
    withterms(MaxWithIndex{colname(column)}(), column)

emptyvalue(::MaxWithIndex{C}) where {C} =
    NamedTuple{(Symbol(C, :_max), Symbol(C, :_maxindex))}((missing, missing))
fresh(::MaxWithIndex{C}, intypes::NamedTuple) where {C} =
    IndexState{C,(Symbol(C, :_max), Symbol(C, :_maxindex)),intypes[C],typeof(max),
        :both}()

"""
    MinWithIndex(column::ColumnSpec) -> Summarizer

The minimum of `column` and how many rows ago it was taken, in `:{column}_min`
(with `column`'s element type) and `:{column}_minindex` (an `Int`), both
`missing` for no rows. It folds one state for both, where [`Min`](@ref) and
[`MinIndex`](@ref) together fold two, and so can't be requested with `Min` of
the same column. The index follows `MinIndex`.

# Arguments
- `column`: a column name, or a row term `name => f` (see [`ColumnSpec`](@ref)).

# Examples
```jldoctest
df = DataFrame(time = 1:6, x = [3, 1, 4, 1, 5, 9])
p = readtable(df) |> summarize(MinWithIndex(:x))
DataFrame(load(Context(0, 10), p))

# output

1×3 DataFrame
 Row │ time   x_min  x_minindex
     │ Int64  Int64  Int64
─────┼──────────────────────────
   1 │    10      1           2
```
"""
struct MinWithIndex{C} <: GroupSummarizer end
MinWithIndex(column::ColumnSpec) =
    withterms(MinWithIndex{colname(column)}(), column)

emptyvalue(::MinWithIndex{C}) where {C} =
    NamedTuple{(Symbol(C, :_min), Symbol(C, :_minindex))}((missing, missing))
fresh(::MinWithIndex{C}, intypes::NamedTuple) where {C} =
    IndexState{C,(Symbol(C, :_min), Symbol(C, :_minindex)),intypes[C],typeof(min),
        :both}()

# Every extreme but Last slides the deque above, since TrackState and
# IndexState have no inverse.
freshwindowed(s::Union{Min,Max,First,MaxIndex,MinIndex,MaxWithIndex,
    MinWithIndex}, intypes::NamedTuple) = windowtrack(fresh(s, intypes))
