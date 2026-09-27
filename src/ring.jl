# A fixed-capacity ring of rows, the one ring buffer in the package: the
# count-window state (barwindow.jl) keeps its window's rows in one, and
# addrollingcolumns keeps one per key for its `Bars` windows. A full ring
# overwrites its oldest row, so a caller that must downdate that row sizes the
# ring one past its window and reads the row back with `ringback` after the
# push. The storage is allocated once; pushing never allocates.
mutable struct RowRing{R}
    const rows::Vector{R}
    first::Int  # index of the oldest row
    len::Int
end

RowRing{R}(capacity::Int) where {R} = RowRing{R}(Vector{R}(undef, capacity), 1, 0)

ringlength(r::RowRing) = r.len

# Append `row`, overwriting the oldest row when the ring is full.
@inline function ringpush!(r::RowRing{R}, row) where {R}
    cap = length(r.rows)
    if r.len < cap
        i = r.first + r.len
        @inbounds r.rows[i > cap ? i - cap : i] = row
        r.len += 1
    else
        @inbounds r.rows[r.first] = row
        r.first = r.first == cap ? 1 : r.first + 1
    end
    return nothing
end

# The j-th newest row (j = 1 is the newest); the caller checks j <= ringlength.
@inline function ringback(r::RowRing, j::Int)
    i = r.first + r.len - j
    cap = length(r.rows)
    return @inbounds r.rows[i > cap ? i - cap : i]
end

emptyring!(r::RowRing) = (r.first = 1; r.len = 0; r)

# The same rows, oldest first, in a new ring of row type `R` and capacity `cap`
# (at least the ring's length); for a widening.
function convertring(::Type{R}, r::RowRing, cap::Int) where {R}
    out = RowRing{R}(cap)
    for j in ringlength(r):-1:1
        ringpush!(out, convert(R, ringback(r, j)))
    end
    return out
end
