# The count-window state: a structured summarizer's states over the last n rows
# folded, for a summarizer whose own state needs a trailing window mid-recursion
# (an indicator's inner moving average, say). The expanded summarizers are
# partitioned by `tiering` as the window transforms do: groups slide windowed
# states, update!d on admission and downdate!d as their row leaves the ring;
# other monoids sit in a two-stack queue of combine!d partial states; fieldless
# dependents read the others through `mergestates` at `value` time. Everything
# is preallocated, so a row costs no allocation.

"""
    barwindow(s::Summarizer, n::Integer, intypes::NamedTuple) -> SummarizerState

A state summarizing only the last `n` rows folded into it, for a summarizer
whose own state needs a trailing window (a moving average inside a recursive
indicator, say). [`update!`](@ref) folds a row in and drops the row `n` rows
older; [`value`](@ref) is `s`'s value over the last `n` rows, with every
column `missing` until `n` rows have arrived, so each value column is
`Union{Missing, T}`.

`s`'s dependencies are expanded and shared as in the summarizing transforms.
Groups slide in O(1) per row and other monoids take O(1) amortized, and neither
allocates. The state's type depends on `s`, `n`'s tier choice and `intypes`,
not on their types alone, so an outer state holding one should take its type
as a type parameter.

# Arguments
- `s`: a [`MonoidSummarizer`](@ref), which may read row terms; any other
  summarizer, or a dependency that is one, is an `ArgumentError`.
- `n`: the window's row count, at least 1.
- `intypes`: the element types of the columns the rows passed to `update!`
  carry, as in [`fresh`](@ref). A row term's type is inferred from them and
  must be concrete, or a `Union` of concrete types; its name may not be one
  of theirs.

```jldoctest
st = CausalFrames.barwindow(Mean(:x), 2, (x = Float64,))
map(1.0:4.0) do x
    CausalFrames.update!(st, (; x))
    CausalFrames.value(st)
end

# output

4-element Vector{@NamedTuple{x_mean::Union{Missing, Float64}}}:
 (x_mean = missing,)
 (x_mean = 1.5,)
 (x_mean = 2.5,)
 (x_mean = 3.5,)
```

Rows leave the window oldest first, so a wrapped [`downdate!`](@ref) sees the
order it expects. [`widenstate`](@ref) re-types the state and keeps its window.
"""
function barwindow(s::Summarizer, n::Integer, intypes::NamedTuple)
    n >= 1 || throw(ArgumentError("barwindow count must be positive, got $n"))
    protos, requested, terms =
        prototypes(Summarizer[s], Symbol[], "barwindow")
    for k in keys(terms)
        haskey(intypes, k) && throw(
            ArgumentError(
                "barwindow row term $(repr(k)) collides with an input column"),
        )
    end
    return newbarwindow(protos, Val(requested), terms, Int(n), intypes)
end

# The row types: rows are stored with their row terms evaluated, so a row is
# downdated with the values it was folded with and no term runs twice.
function barwindowtypes(terms::NamedTuple, intypes::NamedTuple)
    R0 = storerowtype(intypes)
    termtypes = map(keys(terms), values(terms)) do k, f
        T = Base.promote_op(f, R0)
        concreteterm(T) || throw(
            ArgumentError(
                "barwindow row term $(repr(k)) must infer a concrete type, got $T"),
        )
        T
    end
    return merge(intypes, NamedTuple{keys(terms)}(termtypes))
end

concreteterm(T) =
    isconcretetype(T) || (T isa Union && all(isconcretetype, Base.uniontypes(T)))

# Build the state for the realized types; type-unstable, once per state and
# per widening.
function newbarwindow(protos::Tuple, outs::Val{O}, terms::NamedTuple, n::Int,
    intypes::NamedTuple) where {O}
    types = barwindowtypes(terms, intypes)
    tg = tiering(protos, types)
    # A fieldless dependent takes no tier, so this rejects only a summarizer
    # that would need re-folding: a plain one with state.
    for (p, tier) in zip(protos, tiernames(tg))
        tier === :refold && throw(
            ArgumentError(
                "barwindow summarizers must be MonoidSummarizers, got $(typeof(p))",
            ),
        )
    end
    R = storerowtype(types)
    V = missingfields(valuetype(typeof(mergedstates(tg)), outs), outs)
    queue = isempty(tg.tree) ? nothing : BarQueue(tg.tree, n)
    return BarWindowState{R,storerowtype(intypes),O,V}(n, RowRing{R}(n + 1),
        map(fresh, tg.running), queue, tg.derived, tg.perm, protos, terms)
end

# R is the stored row type, R0 the input row type, O the output names, V
# the value type; `running` holds windowed group states, `queue` the other
# monoids (`nothing` when there are none), `derived` the fieldless dependents.
# The ring holds n + 1 rows, so the row leaving is still there to downdate.
mutable struct BarWindowState{R,R0,O,V,SR<:Tuple,Q,SD<:Tuple,P,PR<:Tuple,
    TM<:NamedTuple} <: SummarizerState
    const n::Int
    const ring::RowRing{R}
    running::SR
    const queue::Q
    const derived::SD
    const perm::Val{P}
    const protos::PR   # for widenstate
    const terms::TM
end

BarWindowState{R,R0,O,V}(n::Int, ring::RowRing{R}, running::SR, queue::Q,
    derived::SD, perm::Val{P}, protos::PR,
    terms::TM) where {R,R0,O,V,SR,Q,SD,P,PR,TM} =
    BarWindowState{R,R0,O,V,SR,Q,SD,P,PR,TM}(n, ring, running, queue, derived,
        perm, protos, terms)

# The window's own tuple folds. The transforms' helpers (`updateall!`,
# `summaryvalues`, …) would compute the same, but a count window runs inside an
# outer state's `update!` and `value`, which those same helpers call, and
# inference cuts a repeated caller-to-callee edge short (its recursion limit):
# the inner fold would dispatch and allocate per row. Straight-line generated
# code, one call per state, gives the window edges of its own.
@generated function barupdate!(states::Tuple, row)
    calls = [:(update!(getfield(states, $i), row)) for i in 1:fieldcount(states)]
    return Expr(:block, Expr(:meta, :inline), calls..., nothing)
end

@generated function bardowndate!(states::Tuple, row)
    calls = [:(downdate!(getfield(states, $i), row)) for i in 1:fieldcount(states)]
    return Expr(:block, Expr(:meta, :inline), calls..., nothing)
end

@generated function barfresh!(states::Tuple)
    calls = [:(fresh!(getfield(states, $i))) for i in 1:fieldcount(states)]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, calls...))
end

@generated function barcombine!(dest::S, a::S, b::S) where {S<:Tuple}
    calls = [
        :(combine!(getfield(dest, $i), getfield(a, $i), getfield(b, $i)))
        for i in 1:fieldcount(S)
    ]
    return Expr(:block, Expr(:meta, :inline), calls..., nothing)
end

# `summaryvalues`, one binding per state.
@generated function barvalues(states::Tuple, ::Val{R}) where {R}
    v(i) = Symbol(:vals, i)
    body = Any[Expr(:meta, :inline), :($(v(0)) = (;))]
    for i in 1:fieldcount(states)
        push!(body,
            :($(v(i)) = merge($(v(i - 1)), value(getfield(states, $i), $(v(i - 1))))))
    end
    push!(body, :(NamedTuple{R}($(v(fieldcount(states))))))
    return Expr(:block, body...)
end

# The two-stack queue over the non-group monoids' states: the front stack holds
# suffix combines of an older batch of rows (front[i] folds rows i:nfront of
# it, the live ones being top:nfront), `back` folds the rows since. Popping an
# empty front flips the back rows (still in the ring) into it: one update! and
# one combine! per row, so O(1) amortized per row, and none of it allocates.
mutable struct BarQueue{S<:Tuple}
    const front::Vector{S}
    top::Int
    nfront::Int
    back::S
    single::S  # scratch: one row's states, during a flip
    out::S     # scratch: the value's combine, borrowed until the next one
end

BarQueue(protos::S, n::Int) where {S<:Tuple} =
    BarQueue{S}([map(fresh, protos) for _ in 1:n], 1, 0, map(fresh, protos),
        map(fresh, protos), map(fresh, protos))

@inline queuepush!(::Nothing, row) = nothing
@inline queuepush!(q::BarQueue, row) = barupdate!(q.back, row)

# Drop the oldest row, the window being full with n rows before the one just
# pushed into the ring.
@inline queuepop!(::Nothing, ring::RowRing, n::Int) = nothing
@inline function queuepop!(q::BarQueue, ring::RowRing, n::Int)
    q.top > q.nfront && flipqueue!(q, ring, n)
    q.top += 1
    return nothing
end

# Every window row is in `back` when the front is empty: they are the ring's
# rows before the newest, front[i] taking the i-th oldest.
@noinline function flipqueue!(q::BarQueue, ring::RowRing, n::Int)
    front = q.front
    last = barfresh!(@inbounds front[n])
    barupdate!(last, ringback(ring, 2))
    @inbounds front[n] = last
    for i in (n-1):-1:1
        single = barfresh!(q.single)
        q.single = single
        barupdate!(single, ringback(ring, n + 2 - i))
        @inbounds barcombine!(front[i], single, front[i+1])
    end
    q.back = barfresh!(q.back)
    q.nfront = n
    q.top = 1
    return nothing
end

@inline queuestates(::Nothing) = ()
@inline function queuestates(q::BarQueue)
    q.top > q.nfront && return q.back
    barcombine!(q.out, @inbounds(q.front[q.top]), q.back)
    return q.out
end

queuefresh(::Nothing) = nothing
queuefresh(q::BarQueue) = BarQueue(q.back, length(q.front))

queuefresh!(::Nothing) = nothing
function queuefresh!(q::BarQueue)
    q.back = barfresh!(q.back)
    q.top = 1
    q.nfront = 0
    return nothing
end

# The row as stored: the input columns, projected by name and converted to the
# input types the row terms' types were inferred from, then the row terms.
@inline barrow(st::BarWindowState{R,R0}, row) where {R,R0} =
    termrow(R, projectrow(R0, row), st.terms)

@inline termrow(::Type{R}, base::NamedTuple, terms::NamedTuple{()}) where {R} =
    convert(R, base)
@inline termrow(::Type{R}, base::NamedTuple, terms::NamedTuple) where {R} =
    convert(R, merge(base, map(f -> f(base), terms)))

# Dispatching on the names keeps the projection inferred without relying on
# constant folding; a NamedTuple row takes Base's own projection.
@inline projectrow(::Type{NamedTuple{N,T}}, row::NamedTuple) where {N,T} =
    convert(NamedTuple{N,T}, NamedTuple{N}(row))
@inline projectrow(::Type{NamedTuple{N,T}}, row) where {N,T} =
    NamedTuple{N,T}(map(c -> getproperty(row, c), N))

function update!(st::BarWindowState, row)
    admitbar!(st, barrow(st, row))
    return nothing
end

# Admit a stored row: push it, fold it into the groups, then (with the window
# overfull) pop the queue's oldest before pushing, and downdate the groups'
# oldest, still in the ring.
@inline function admitbar!(st::BarWindowState{R}, row::R) where {R}
    ring = st.ring
    ringpush!(ring, row)
    barupdate!(st.running, row)
    full = ringlength(ring) > st.n
    full && queuepop!(st.queue, ring, st.n)
    queuepush!(st.queue, row)
    full && bardowndate!(st.running, ringback(ring, st.n + 1))
    return nothing
end

function value(st::BarWindowState{R,R0,O,V}) where {R,R0,O,V}
    ringlength(st.ring) < st.n && return missingrow(V, Val(O))
    states = mergestates(st.perm,
        (st.running, queuestates(st.queue), (), st.derived))
    return convert(V, barvalues(states, Val(O)))
end

fresh(st::BarWindowState{R,R0,O,V}) where {R,R0,O,V} =
    BarWindowState{R,R0,O,V}(st.n, RowRing{R}(st.n + 1), map(fresh, st.running),
        queuefresh(st.queue), st.derived, st.perm, st.protos, st.terms)

function fresh!(st::BarWindowState)
    emptyring!(st.ring)
    st.running = barfresh!(st.running)
    queuefresh!(st.queue)
    return st
end

# Re-tier for the wider types (a state that stops being invertible moves to the
# queue) and replay the window's rows.
function widenstate(st::BarWindowState{R,R0,O}, intypes::NamedTuple) where {R,R0,O}
    w = newbarwindow(st.protos, Val(O), st.terms, st.n, intypes)
    replaybar!(w, st.ring, st.n)
    return w
end

function replaybar!(w::BarWindowState{R}, ring::RowRing, n::Int) where {R}
    for j in min(ringlength(ring), n):-1:1
        admitbar!(w, convert(R, ringback(ring, j)))
    end
    return w
end
