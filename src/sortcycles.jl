# sortcycles: a stable reordering of the rows within each cycle (a maximal run of
# rows sharing one timestamp). The time order is untouched, so the output obeys
# the chunk protocol by construction; the only state is the latest cycle, held
# back because the rest of it may arrive in the next chunk.

"""
    sortcycles(by; rev = false) -> (CausalPipeline -> CausalPipeline)
    sortcycles(p::CausalPipeline, by; rev = false) -> CausalPipeline

A transform reordering the rows within each *cycle* — a maximal run of rows
sharing one timestamp — while leaving the time order and every column as they
are. `by` chooses the order:

- a column name (a `Symbol` or `AbstractString`);
- a collection of column names, compared lexicographically;
- a per-row function, receiving the same map-like row object as
  [`filterrows`](@ref) and returning the row's sort key (return a tuple to
  compare several values lexicographically).

The sort is stable, so rows with equal keys keep their stream order. Keys are
compared with `isless`, so `missing` sorts last — or first with `rev = true`,
which reverses the whole order. For a mixed order, negate a numeric key in a
function: `sortcycles(r -> (-r.votes, r.id))`.

This is the within-timestamp half of an SQL `ORDER BY time, ...`, which is what
makes a keyed [`Count`](@ref) over `key = :time` a rank:

```julia
readtable(films; time = :year) |>
    sortcycles(r -> (-r.votes, r.id)) |>
    addsummarycolumns(Count(); key = :time)   # :count ranks films within a year
```

`sortcycles` is causal: a cycle is reordered using only its own rows. It holds
back the latest cycle until a row with a later time arrives or the stream ends,
so memory is one cycle plus one chunk. Splitting the context changes nothing,
since every row at a split time falls in the later half.

Naming a column the data does not have is an `ArgumentError` when the pipeline
runs; an empty collection of names, or a `by` that is neither a name, a
collection of names nor a function, is an `ArgumentError` at construction.

The curried form composes with `|>`; the uncurried form applies directly, so
`sortcycles(p, by)` is equivalent to `p |> sortcycles(by)`.
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
    ArgumentError("sortcycles: expected a column name, a collection of column \
        names, or a per-row function, got $(typeof(x))"))

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
