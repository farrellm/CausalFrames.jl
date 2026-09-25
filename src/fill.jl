# The missing-value fills. `fillmissing` is row-wise and stateless — a constant
# per column, so a chunk is all the context it needs. `forwardfill` carries the
# last non-missing value of every filled column across rows, chunks and keys,
# which makes it stateful in the sense DESIGN.md's streaming section means: the
# carried state is what keeps `stream` equal to `load`.
#
# That state is a cell per (key, column), not a row per key, because a forward
# fill is column-independent — `:a` may carry from row 3 while `:b` carries from
# row 7 — so `lastrow`'s whole-row store cannot serve it. The cells are mutable,
# so the store is a plain `Dict{K,NamedTuple}` rather than join.jl's
# `Dict{K,Int}` over a slot vector: a lookup already answers with a pointer, so
# there is no `Union{Nothing,V}` to box (`KeyBuffer`'s reason, src/acausal.jl).

"""
    forwardfill(selectors...; key = nothing, tolerance = nothing)
        -> (CausalPipeline -> CausalPipeline)
    forwardfill(p::CausalPipeline, selectors...; key = nothing,
                tolerance = nothing) -> CausalPipeline

A transform replacing `missing` in the selected columns with the column's last
non-missing value. Each column is filled independently. Rows before a column's
first value, or past `tolerance`, stay `missing`. To fill with a constant, use
[`fillmissing`](@ref).

# Arguments
- `selectors`: the columns to fill, as for [`selectcolumns`](@ref). `:time` and
  key columns are never filled. The set of selected columns may not change
  between chunks.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. With a key, values carry only within the same key.
- `tolerance = nothing`: the maximum age of a carried value, measured from the
  row it came from; must be non-negative. The input then runs over
  `[start - tolerance, stop)`, so rows near `start` can be filled from before
  the window; the time type must support subtraction.

```jldoctest
using Dates
t0 = DateTime(2026, 1, 1)
df = DataFrame(time = t0 .+ Minute.([0, 1, 2, 10]), sym = ["a", "b", "a", "a"],
               bid = [1.0, 2.0, missing, missing])
p = readtable(df) |> forwardfill(:bid; key = :sym, tolerance = Minute(5))
DataFrame(load(Context(t0, t0 + Hour(1)), p))

# output

4×3 DataFrame
 Row │ time                 sym     bid
     │ DateTime             String  Float64?
─────┼────────────────────────────────────────
   1 │ 2026-01-01T00:00:00  a             1.0
   2 │ 2026-01-01T00:01:00  b             2.0
   3 │ 2026-01-01T00:02:00  a             1.0
   4 │ 2026-01-01T00:10:00  a       missing
```
"""
function forwardfill(selectors...; key = nothing, tolerance = nothing)
    checkselectors(selectors, "forwardfill", false)
    keycols = keycolumns(key, "forwardfill")
    keynames = Val(Tuple(keycols))
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            st = ForwardFillState(keycols, selectors)
            step = c -> fillchunk!(st, keynames, tolerance, ctx.start, c)
            return chunkmap(step, p.run(fillcontext(ctx, tolerance)))
        end
    end
end
forwardfill(p::CausalPipeline, selectors...; kwargs...) =
    forwardfill(selectors...; kwargs...)(p)

# The mirror of asofjoin's `rightcontext`: non-negativity is probed by
# subtracting rather than by comparing against zero, so a time type with no
# zero of its own (a `Period`, say) still works.
fillcontext(ctx::Context, ::Nothing) = ctx
function fillcontext(ctx::Context, tolerance)
    start = ctx.start - tolerance
    start <= ctx.start || throw(ArgumentError(
        "forwardfill tolerance must be non-negative, got $tolerance"))
    return Context(start, ctx.stop)
end

# One carried value per (key, filled column): the last non-missing value, the
# time of the row it came from, and whether there has been one. Mutable, so a
# store lookup answers with a pointer and updates in place. `seen` leads so the
# inner constructor can leave `value` and `time` undefined — TrackState's idiom
# (src/summarizers.jl) — and it guards every read of them. Both parameters are
# bound by the constructor's own signature, so Aqua's unbound-parameter check
# needs nothing further.
mutable struct FillCell{T,S}
    seen::Bool
    value::T
    time::S
    FillCell{T,S}() where {T,S} = new{T,S}(false)
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state:
# which columns are filled comes from the first chunk's names and the cell
# types from the promoted schema, so neither is known until a chunk arrives,
# and everything per-row sits behind the `fillkeyless!`/`fillkeyed!` barriers.
mutable struct ForwardFillState
    const keycols::Vector{Symbol}
    const selectors::Tuple
    lastnames::Union{Nothing,Vector{String}}  # names the fill set was resolved from
    fillnames::Union{Nothing,Vector{Symbol}}  # the columns being filled
    fillval::Any     # Val{Tuple(fillnames)}, memoized alongside it
    types::Union{Nothing,NamedTuple}          # promotion of every schema seen
    cells::Any       # keyless: the NamedTuple of cells; keyed: Dict{K,NT} of them
    passedstart::Bool                         # a row at or after ctx.start seen
end
ForwardFillState(keycols::Vector{Symbol}, selectors::Tuple) =
    ForwardFillState(keycols, selectors, nothing, nothing, nothing, nothing,
        nothing, false)

function fillchunk!(st::ForwardFillState, keynames::Val, tolerance, start,
    c::DataFrame)
    resolvefill!(st, c)
    types = promotetypes(st.types, chunktypes(c))
    widened = st.types !== nothing && types != st.types
    st.types = types
    fillprepared!(st, keynames, st.fillval, tolerance, types, widened, c)
    return clipstart!(st, c, start)
end

# The columns to fill, in the chunk's own order. Running the selectors over
# every column of every chunk is wasted work when the schema never moves, so
# the resolution is memoized against the names it came from, as
# `selectcolumns` memoizes its own projection. Element types may move from
# chunk to chunk, but the set of filled columns may not: it fixes the cell
# NamedTuple's names, and hence the store's type.
function resolvefill!(st::ForwardFillState, c::DataFrame)
    cols = names(c)
    st.lastnames == cols && return nothing
    checkkeycolumns(st.keycols, c, "forwardfill")
    foreachliteral(st.selectors) do n
        n in cols ||
            throw(ArgumentError("forwardfill: no column named $(repr(Symbol(n)))"))
    end
    fills = Symbol[]
    for n in cols
        (n == "time" || Symbol(n) in st.keycols) && continue
        matchescolumn(st.selectors, n) && push!(fills, Symbol(n))
    end
    st.fillnames === nothing || fills == st.fillnames ||
        throw(
            ArgumentError("forwardfill: the columns to fill changed mid-stream, \
                from $(st.fillnames) to $(fills)"),
        )
    st.fillnames = fills
    st.fillval = Val(Tuple(fills))
    st.lastnames = cols
    return nothing
end

# A cell holds only non-missing values, so it is typed at the non-missing type.
# The `Union{}` guard keeps a pathological all-Missing column working: there is
# then no non-missing type to hold, and `seen` simply never becomes true.
function cellvaluetype(T::Type)
    V = nonmissingtype(T)
    return V === Union{} ? T : V
end

cellstype(types::NamedTuple, ::Val{FN}) where {FN} = NamedTuple{FN,
    Tuple{(FillCell{cellvaluetype(types[n]),types.time} for n in FN)...}}

@inline freshcells(::Type{NamedTuple{FN,TT}}) where {FN,TT} =
    NamedTuple{FN,TT}(ntuple(i -> fieldtype(TT, i)(), Val(length(FN))))

# Widening copies through the field assignments, which convert, and skips
# unseen cells — their value and time are deliberately undefined.
@inline function widencell(::Type{C}, old::FillCell) where {C<:FillCell}
    cell = C()
    if old.seen
        cell.value = old.value
        cell.time = old.time
        cell.seen = true
    end
    return cell
end

widencells(::Type{NamedTuple{FN,TT}}, old::NamedTuple{FN}) where {FN,TT} =
    NamedTuple{FN,TT}(ntuple(i -> widencell(fieldtype(TT, i), old[i]), Val(length(FN))))

function widenstore(::Type{Dict{K,NT}}, old::AbstractDict) where {K,NT}
    new = Dict{K,NT}()
    sizehint!(new, length(old))
    for (k, v) in old
        new[convert(K, k)] = widencells(NT, v)
    end
    return new
end

# Per-chunk setup: build or widen the cells, allocate the replacement columns,
# then hand concretely typed arguments to the per-row kernel. A column whose
# promoted type admits no `Missing` has nothing to fill, so it gets no
# replacement column at all and is left untouched — `nothing` in the group
# tuple, which folds the write away in the kernel.
function fillprepared!(st::ForwardFillState, ::Val{KN}, ::Val{FN}, tolerance,
    types::NamedTuple, widened::Bool, c::DataFrame) where {KN,FN}
    isempty(FN) && return nothing
    NT = cellstype(types, Val(FN))
    nt = Tables.columntable(c)
    outs = filloutputs(types, Val(FN), nrow(c))
    groups = map(n -> (getproperty(nt, n), outs[n]), FN)
    if isempty(KN)
        cells =
            st.cells === nothing ? freshcells(NT) :
            widened ? widencells(NT, st.cells) : st.cells
        st.cells = cells
        fillkeyless!(nt.time, groups, values(cells::NT), tolerance)
    else
        D = Dict{storekeytype(types, Val(KN)),NT}
        store = st.cells === nothing ? D() :
                widened ? widenstore(D, st.cells) : st.cells
        st.cells = store
        fillkeyed!(nt.time, groups, nt, Val(KN), store, tolerance)
    end
    attach!(c, outs, Val(FN))
    return nothing
end

# The replacement column takes the promoted input type, so a column that has
# admitted `Missing` in any chunk keeps admitting it — the leading rows before
# the first value, and the rows past `tolerance`, really are missing.
filloutputs(types::NamedTuple, ::Val{FN}, n::Int) where {FN} =
    NamedTuple{FN}(map(k -> Missing <: types[k] ? Vector{types[k]}(undef, n) :
                            nothing, FN))

function attach!(c::DataFrame, outs::NamedTuple{FN}, ::Val{FN}) where {FN}
    for n in FN
        col = outs[n]
        # Columns are replaced wholesale, never mutated in place; assigning an
        # existing name keeps its position, so the output schema is the input's.
        col === nothing || (c[!, n] = col)
    end
    return nothing
end

# --- fill kernels ----------------------------------------------------------
#
# Called with concretely typed arguments, so the per-row work compiles to
# direct column access with nothing boxed. `groups` is a tuple of (input
# column, replacement column) pairs and `cells` the matching cells, both in the
# filled columns' own order, so the walk over the columns unrolls and no name
# is a runtime value.

function fillkeyless!(times::AbstractVector, groups::Tuple, cells::Tuple, tolerance)
    for i in eachindex(times)
        fillrow!(i, @inbounds(times[i]), tolerance, groups, cells)
    end
    return nothing
end

function fillkeyed!(times::AbstractVector, groups::Tuple, nt::NamedTuple,
    ::Val{KN}, store::Dict{K,NT}, tolerance) where {KN,K,NT}
    for i in eachindex(times)
        cells = get!(() -> freshcells(NT), store, keyat(nt, i, Val(KN)))
        fillrow!(i, @inbounds(times[i]), tolerance, groups, values(cells))
    end
    return nothing
end

@inline fillrow!(::Int, _, _, ::Tuple{}, ::Tuple{}) = nothing
@inline function fillrow!(i::Int, t, tolerance, groups::Tuple, cells::Tuple)
    incol, outcol = groups[1]
    cell = cells[1]
    x = @inbounds incol[i]
    if ismissing(x)
        # A carried value keeps the time of the row it came from, so staleness
        # is decided here, against each row — never by evicting eagerly, which
        # is asofjoin's rule too.
        outcol === nothing || (@inbounds outcol[i] =
            cell.seen && (tolerance === nothing || t - cell.time <= tolerance) ?
            cell.value : missing)
    else
        cell.value = x
        cell.time = t
        cell.seen = true
        outcol === nothing || (@inbounds outcol[i] = x)
    end
    return fillrow!(i, t, tolerance, Base.tail(groups), Base.tail(cells))
end

# With `tolerance` the input ran over the widened context, so the leading rows
# fill the cells and are then dropped: `load` rejects a chunk starting before
# `ctx.start`. Times are non-decreasing across chunks, so the first chunk with
# an in-window row settles it for every later one.
function clipstart!(st::ForwardFillState, c::DataFrame, start)
    st.passedstart && return c
    lo = searchsortedfirst(c.time, start)
    lo > nrow(c) && return nothing
    st.passedstart = true
    return lo == 1 ? c : c[lo:end, :]
end

# --- fillmissing -----------------------------------------------------------

"""
    fillmissing(specs...) -> (CausalPipeline -> CausalPipeline)
    fillmissing(p::CausalPipeline, specs...) -> CausalPipeline

A transform replacing `missing` in the named columns with a constant per
column. A filled column's element type becomes
`promote_type(nonmissingtype(T), typeof(value))`; a column that cannot hold
`missing` is left alone. To carry the last value forward, use
[`forwardfill`](@ref).

# Arguments
- `specs`: at least one fill, as `name => value` pairs, a `NamedTuple`, or a
  collection of pairs. Names must be unique, may not be `time`, and must exist
  in the data (checked when a chunk arrives).

```jldoctest
df = DataFrame(time = [1, 2], qty = [missing, 3.0], sym = ["a", missing])
p = readtable(df) |> fillmissing(:qty => 0.0, :sym => "")   # or ((qty = 0.0, sym = ""))
DataFrame(load(Context(0, 10), p))

# output

2×3 DataFrame
 Row │ time   qty      sym
     │ Int64  Float64  String
─────┼────────────────────────
   1 │     1      0.0  a
   2 │     2      3.0
```
"""
function fillmissing(specs...)
    fillnames, vals = tofillvalues(specs)
    isempty(fillnames) &&
        throw(ArgumentError("fillmissing requires at least one column => value pair"))
    allunique(fillnames) ||
        throw(ArgumentError("fillmissing column names must be unique"))
    :time in fillnames &&
        throw(ArgumentError("fillmissing may not fill the time column"))
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> fillmissingchunk!(c, fillnames, vals), p.run(ctx))
        end
    end
end
fillmissing(p::CausalPipeline, specs...) = fillmissing(specs...)(p)

# Parallel tuples — names as a Symbol tuple, values as a possibly
# heterogeneous one — so the fill values keep their concrete types into the
# per-column kernel. `towindows` (src/rolling.jl) has the same shape and the
# same reason. The two leading methods are the inferrable spellings; the
# fallback flattens whatever collection of pairs was passed.
tofillvalues(specs::Tuple{NamedTuple}) = (keys(specs[1]), values(specs[1]))
tofillvalues(specs::Tuple{Vararg{Pair}}) =
    (map(p -> Symbol(first(p)), specs), map(last, specs))
function tofillvalues(specs::Tuple)
    ps = Pair[]
    for s in specs
        if s isa Pair
            push!(ps, s)
        elseif applicable(iterate, s)
            all(p -> p isa Pair, s) || fillspecerror(s)
            append!(ps, s)
        else
            fillspecerror(s)
        end
    end
    return (Tuple(Symbol(first(p)) for p in ps), Tuple(last(p) for p in ps))
end

fillspecerror(x) = throw(
    ArgumentError(
        "invalid fillmissing spec of type $(typeof(x)): expected a NamedTuple, a \
        name => value pair, or a collection of those"),
)

function fillmissingchunk!(c::DataFrame, fillnames::Tuple, vals::Tuple)
    for (n, v) in zip(fillnames, vals)
        hasproperty(c, n) ||
            throw(ArgumentError("fillmissing: no column named $(repr(n))"))
        col = c[!, n]
        # Nothing to replace, so the column passes through untouched.
        Missing <: eltype(col) || continue
        c[!, n] = fillcolumn(col, v)
    end
    return c
end

# Function barrier: with the column dispatched concretely the output element
# type is a compile-time constant, so the comprehension builds a typed column
# directly. `nonmissingtype(T) === Union{}` (an all-Missing column) falls out
# on its own — `promote_type(Union{}, V)` is `V`.
fillcolumn(col::AbstractVector{T}, v::V) where {T,V} =
    promote_type(nonmissingtype(T), V)[ismissing(x) ? v : x for x in col]
