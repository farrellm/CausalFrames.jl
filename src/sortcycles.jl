# sortcycles: a stable reordering of the rows within each cycle (a maximal run of
# rows sharing one timestamp). The time order is untouched, so the output obeys
# the chunk protocol by construction; the only state is the latest cycle, held
# back because the rest of it may arrive in the next chunk.

"""
    sortcycles(by; rev = false) -> (CausalPipeline -> CausalPipeline)
    sortcycles(p::CausalPipeline, by; rev = false) -> CausalPipeline

A transform stably sorting the rows within each *cycle* — a run of rows sharing
one time — leaving the time order untouched. It holds back only the latest
cycle, so memory is one cycle plus one chunk.

# Arguments
- `by`: the sort key — a column name, a non-empty collection of column names
  (compared lexicographically), or a function `row -> key`, with `row` as for
  [`filterrows`](@ref). Keys compare with `isless`, so `missing` sorts last.
  Anything else is an `ArgumentError`, as is naming a column the data lacks.

# Keywords
- `rev = false`: reverse the order. For a mixed order, negate a numeric key in
  a function instead: `sortcycles(r -> (-r.votes, r.id))`.

Sorting cycles turns a keyed [`Count`](@ref) over `key = :time` into a rank:

```julia
readtable(films; time = :year) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :time)   # :count ranks films within a year
```
"""
function sortcycles(by; rev::Bool = false)
    spec = sortspec(by)
    # A concrete ordering rather than a runtime Bool, so the sort inside the
    # barrier specializes instead of splitting on the direction per comparison.
    order = rev ? Base.Order.Reverse : Base.Order.Forward
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            st = CycleSortState()
            return chunkmap(c -> sortcyclechunk!(st, spec, order, c), p.run(ctx);
                flush = () -> flushcycle!(st, spec, order))
        end
    end
end
sortcycles(p::CausalPipeline, by; kwargs...) = sortcycles(by; kwargs...)(p)

# Eager validation: a function, a name, or a non-empty collection of names,
# normalized to a tuple of Symbols. A CausalPipeline lands in the error too,
# which turns a mistyped `sortcycles(p)` into a message.
sortspec(f::Function) = f
sortspec(name::Union{Symbol,AbstractString}) = (Symbol(name),)
function sortspec(names)
    applicable(iterate, names) || sortspecerror(names)
    cols = Symbol[]
    for n in names
        n isa Union{Symbol,AbstractString} || sortspecerror(n)
        push!(cols, Symbol(n))
    end
    isempty(cols) &&
        throw(ArgumentError("sortcycles requires at least one column name"))
    return Tuple(cols)
end

sortspecerror(x) = throw(
    ArgumentError("invalid sortcycles spec of type $(typeof(x)): expected a \
        column name, a collection of column names, or a per-row function"))

# Per-run state: the pieces of the open cycle, all sharing its time. Pieces are
# concatenated once, when the cycle closes, so a cycle spread over many chunks
# costs O(rows) rather than a re-concatenation per chunk.
struct CycleSortState
    pending::Vector{DataFrame}
end
CycleSortState() = CycleSortState(DataFrame[])

# Emit every cycle known to be complete, sorted, and hold back the trailing one.
# The emitted rows — the open cycle this chunk closes, then the chunk's own
# complete cycles — are gathered as views and materialized in one copy: nearly
# every chunk closes the previous chunk's tail, so concatenating the tail onto
# the whole chunk first would copy each chunk twice.
function sortcyclechunk!(st::CycleSortState, spec, order, c::DataFrame)
    pending = st.pending
    times = c.time
    lo = searchsortedfirst(times, last(times))   # where the trailing cycle starts
    hi = 0                                        # c's rows closing the open cycle
    closed = nothing
    if !isempty(pending)
        names(c) == names(first(pending)) ||
            throw(ArgumentError("sortcycles: chunk columns changed mid-cycle, \
                from $(names(first(pending))) to $(names(c))"))
        opentime = last(first(pending).time)
        # Times are non-decreasing, so a chunk ending at the open time is
        # entirely that cycle; the chunk is owned, so it is held uncopied.
        if last(times) == opentime
            push!(pending, c)
            return nothing
        end
        hi = searchsortedlast(times, opentime)
        cycle =
            hi == 0 ?
            (length(pending) == 1 ? only(pending) : reduce(vcat, pending)) :
            reduce(vcat, [pending; view(c, 1:hi, :)])
        empty!(pending)
        closed = sortedview(cycle, 1:nrow(cycle), spec, order)
    end
    rest = lo > hi + 1 ? sortedview(c, (hi+1):(lo-1), spec, order) : nothing
    push!(pending, lo == 1 ? c : c[lo:end, :])
    return materialize(closed, rest)
end

materialize(::Nothing, ::Nothing) = nothing
materialize(v::SubDataFrame, ::Nothing) = DataFrame(v)
materialize(::Nothing, v::SubDataFrame) = DataFrame(v)
materialize(a::SubDataFrame, b::SubDataFrame) = vcat(a, b)

# Called once, when upstream is exhausted: the open cycle is complete, and a
# cycle already in order goes out as the pieces' concatenation, uncopied again.
function flushcycle!(st::CycleSortState, spec, order)
    isempty(st.pending) && return nothing
    cycle = length(st.pending) == 1 ? only(st.pending) : reduce(vcat, st.pending)
    empty!(st.pending)
    perm = sortedperm(cycle, 1:nrow(cycle), spec, order)
    return isempty(perm) ? cycle : cycle[perm, :]
end

# `rows` of `df` in sorted order, as a view.
function sortedview(df::DataFrame, rows::UnitRange{Int}, spec, order)
    perm = sortedperm(df, rows, spec, order)
    return isempty(perm) ? view(df, rows, :) : view(df, perm, :)
end

# The type-unstable setup, once per emission: resolve the keys over `rows` and
# hand them to the typed barrier. Returns the rows' permutation as indices into
# `df`, or an empty vector when every cycle was already in order.
function sortedperm(df::DataFrame, rows::UnitRange{Int}, spec, order)
    perm = cycleperm!(Int[], sortkeys(spec, df, rows), view(df.time, rows), order)
    perm .+= first(rows) - 1
    return perm
end

function sortkeys(names::Tuple{Vararg{Symbol}}, df::DataFrame,
    rows::UnitRange{Int})
    for n in names
        columnindex(df, n) > 0 ||
            throw(ArgumentError("sortcycles: no column named $(repr(n))"))
    end
    return map(n -> view(df[!, n], rows), names)
end
# Only over `rows`, so the held-back cycle is not keyed twice.
sortkeys(f::Function, df::DataFrame, rows::UnitRange{Int}) =
    (maptime(f, map(v -> view(v, rows), Tables.columntable(df))),)

# Function barrier, specialized on the concrete key columns and ordering: walk
# the cycles of `times`, check each for order in one pass, and stably sort the
# stretch of the permutation covering any that is not, comparing the key tuples
# read straight from the columns. `perm` arrives empty and is filled with the
# identity only once a cycle is found out of order, so an in-order stream
# allocates nothing here. Returns `perm`, empty or of length `length(times)`.
function cycleperm!(perm::Vector{Int}, keys::Tuple, times::AbstractVector,
    order::Base.Order.Ordering)
    key = i -> map(k -> @inbounds(k[i]), keys)
    n = length(times)
    a = 1
    @inbounds while a <= n
        t = times[a]
        b = a
        while b < n && times[b+1] == t
            b += 1
        end
        # Stretches before the first sort are still the identity, so the order
        # check reads the range itself rather than the permutation.
        if b > a && !issorted(a:b; by = key, order = order)
            isempty(perm) && append!(perm, 1:n)
            sort!(view(perm, a:b); by = key, order = order,
                alg = Base.Sort.DEFAULT_STABLE)
        end
        a = b + 1
    end
    return perm
end
