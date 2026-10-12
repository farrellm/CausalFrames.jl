# The count-window state: a structured summarizer's states over the last n rows
# folded, for a summarizer whose own state needs a trailing window mid-recursion
# (an indicator's inner moving average, say). The expanded summarizers are
# partitioned by `tiering` as the window transforms do: groups slide windowed
# states, update!d on admission and downdate!d as their row leaves the ring;
# other monoids sit in a two-stack queue of combine!d partial states; fieldless
# dependents read the others through `mergestates` at `value` time. Everything
# is preallocated, so a row costs no allocation.

"""
    barwindow(s::Summarizer, n::Integer, intypes::NamedTuple;
              hoistedby::Union{Nothing, Summarizer} = nothing) -> SummarizerState

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

# Keywords
- `hoistedby = nothing`: the outer summarizer whose state holds the window,
  when it hoists `s`'s row terms: they are then already evaluated, as columns
  of the rows passed to `update!`. It hoists them by returning
  [`barterms`](@ref) from its [`rowterms`](@ref), so the summarizing transforms
  evaluate them over each chunk before storing its rows; the window then reads
  each term from the reserved column `barterms` names it by, and a term whose
  column is missing from `intypes` is an `ArgumentError`. The transforms hold
  a hoisted term to the same typing rule as an unhoisted one. A column of the
  term's own name is never read in its place. With `nothing`, the window
  evaluates the terms itself.

```jldoctest
st = CausalFrames.barwindow(Mean(:x), 2, (x = Float64,))
map(1.0:4.0) do x
    CausalFrames.update!(st, (; x))
    CausalFrames.value(st).x_mean
end

# output

4-element Vector{Union{Missing, Float64}}:
  missing
 1.5
 2.5
 3.5
```

Rows leave the window oldest first, so a wrapped [`downdate!`](@ref) sees the
order it expects. [`widenstate`](@ref) re-types the state and keeps its window.
"""
function barwindow(s::Summarizer, n::Integer, intypes::NamedTuple;
    hoistedby::Union{Nothing,Summarizer} = nothing)
    n >= 1 || throw(ArgumentError("barwindow count must be positive, got $n"))
    protos, requested, terms =
        prototypes(Summarizer[s], Symbol[], "barwindow")
    return newbarwindow(protos, Val(requested), windowterms(terms, hoistedby),
        Int(n), intypes)
end

windowterms(terms::NamedTuple, ::Nothing) = terms
windowterms(terms::NamedTuple, owner::Summarizer) =
    HoistedTerms{keys(terms),hoistednames(owner, keys(terms))}()

# A window's hoisted row terms, by name: columns of its rows, already
# evaluated under the names H.
struct HoistedTerms{K,H} end

termnames(terms::NamedTuple) = keys(terms)
termnames(::HoistedTerms{K}) where {K} = K

# The row terms against the input types, at construction and on every
# widening (a later chunk may add a column of a term's name).
function checkbarterms(terms, intypes::NamedTuple)
    for k in termnames(terms)
        haskey(intypes, k) && throw(
            ArgumentError(
                "barwindow row term $(repr(k)) collides with an input column"),
        )
    end
    return checkhoisted(terms, intypes)
end

checkhoisted(::NamedTuple, ::NamedTuple) = nothing
function checkhoisted(::HoistedTerms{K,H}, intypes::NamedTuple) where {K,H}
    for (k, h) in zip(K, H)
        haskey(intypes, h) || throw(
            ArgumentError(
                "barwindow hoisted row term $(repr(k)) not found in the input; " *
                "return `barterms(s; hoistedby)` from the outer summarizer's " *
                "`rowterms`",
            ),
        )
    end
    return nothing
end

# The columns hoisted row terms are evaluated into: names of their own, so that
# no input column of a term's name can stand in for it, as one would if a
# wrapper left out `barterms`, and scoped by the outer summarizer (its first
# output name, unique within a call), so that two outer summarizers' windows
# never share a term namespace: `#barterm#lastn#d` for `:lastn`'s term `:d`.
hoistednames(owner::Summarizer, ks::Tuple{Vararg{Symbol}}) =
    map(k -> Symbol("#barterm#", first(keys(emptyvalue(owner))), "#", k), ks)

"""
    barterms(s::Summarizer; hoistedby::Summarizer) -> NamedTuple

The row terms a [`barwindow`](@ref) over `s` reads, from `s` and its
dependencies at any depth, for `hoistedby` to hoist. A summarizer whose state
folds rows through a `barwindow` returns them from its own
[`rowterms`](@ref): the summarizing transforms then evaluate them with their
own row terms, and the window is built with the same `hoistedby`. Each term is
keyed by a reserved column name scoped by `hoistedby`'s first output column,
`#barterm#lastn#d` for a term `:d` hoisted by a summarizer producing `:lastn`.
So no input column of the term's name stands in for it, and two summarizers'
windows may each have a term of one name. Each function comes wrapped, so the
transforms hold the term to the window's typing rule: one whose type doesn't
infer as concrete, or a `Union` of concrete types, is an `ArgumentError` on
the first chunk, as it would be unhoisted.

# Arguments
- `s`: the summarizer the window folds, as passed to `barwindow`.

# Keywords
- `hoistedby`: the summarizer whose `rowterms` returns the terms, usually the
  one calling `barterms`. It is required.

```jldoctest
struct LastSum <: CausalFrames.Summarizer end
CausalFrames.emptyvalue(::LastSum) = (; lastsum = missing)
double = r -> 2r.x
terms = CausalFrames.barterms(Sum(:d => double); hoistedby = LastSum())
keys(terms), only(terms).f === double

# output

((Symbol("#barterm#lastsum#d"),), true)
```
"""
function barterms(s::Summarizer; hoistedby::Summarizer)
    terms = prototypes(Summarizer[s], Symbol[], "barterms")[3]
    return NamedTuple{hoistednames(hoistedby, keys(terms))}(
        map((k, f) -> BarTerm{k}(f), keys(terms), values(terms)))
end

# A hoisted row term as `barterms` hands it to the transforms: the window's
# function `f`, under the term's own name `K`. Marked so that `termvalues`
# holds it to the window's rule where `f` is known, rejecting a term that
# doesn't infer on the first chunk, as an unhoisted window does at
# construction, instead of typing it by its values.
struct BarTerm{K,F}
    f::F
end
BarTerm{K}(f::F) where {K,F} = BarTerm{K,F}(f)
(t::BarTerm)(row) = t.f(row)

function termvalues(t::BarTerm{K}, nt::NamedTuple) where {K}
    checktermtype(K, termtype(t.f, termrowtype(typeof(nt))))
    return termvalues(t.f, nt)
end

"""
    barinputcolumns(s::Summarizer;
                    hoistedby::Union{Nothing, Summarizer} = nothing)
        -> Union{Nothing, Tuple{Vararg{Symbol}}}

The input columns a [`barwindow`](@ref) over `s` reads: the union of
[`inputcolumns`](@ref) over `s` and its dependencies, expanded as `barwindow`
expands them. A summarizer whose state folds rows through a `barwindow`
returns it from its own `inputcolumns`, so the window transforms keep only
those columns. It is `nothing`, any column, when one of them may read any
column, or when `s` reads a row term that is not hoisted: the window evaluates
the term from the rows it is given, and which columns the term's function
reads is unknown. A hoisted term is a column of those rows, named in the
result by its reserved name (see [`barterms`](@ref)), beside its own name:
an input column of that name is kept to be reported as a collision.

# Arguments
- `s`: the summarizer the window folds, as passed to `barwindow`.

# Keywords
- `hoistedby = nothing`: the summarizer hoisting `s`'s row terms, as
  `barwindow`'s keyword of the same name.

```jldoctest
struct LastCov <: CausalFrames.Summarizer end
CausalFrames.emptyvalue(::LastCov) = (; lastcov = missing)
term = Covariance(:x, :d => r -> 2r.x)
(CausalFrames.barinputcolumns(Covariance(:x, :y)),
    CausalFrames.barinputcolumns(term),
    CausalFrames.barinputcolumns(term; hoistedby = LastCov()))

# output

((:x, :y), nothing, (:d, :x, Symbol("#barterm#lastcov#d")))
```
"""
function barinputcolumns(s::Summarizer;
    hoistedby::Union{Nothing,Summarizer} = nothing)
    protos, _, terms = prototypes(Summarizer[s], Symbol[], "barinputcolumns")
    cols = readcolumns(protos)
    (cols === nothing || isempty(terms)) && return cols
    hoistedby === nothing && return nothing
    # A term's own name stays, so that a column of the name reaches the window
    # to be reported, whichever transform runs it.
    return Tuple(union(cols, keys(terms), hoistednames(hoistedby, keys(terms))))
end

# The input row's columns and the stored row's: rows are stored with their row
# terms evaluated, so a row is downdated with the values it was folded with and
# no term runs twice. Once a row's terms are evaluated nothing but `protos`
# reads it, so besides the terms only `:time` (which `inputcolumns` leaves out,
# though `update!` may read it) and the columns they read are stored. The two
# methods differ only in where a term's value comes from: evaluated from the
# whole input row, or (hoisted) read under its hoisted name and stored under
# its own. With no terms, the row is cut on input.
function barwindowtypes(::NamedTuple{()}, intypes::NamedTuple, protos::Tuple)
    base = storedbase(intypes, protos, ())
    return base, base
end

function barwindowtypes(terms::NamedTuple, intypes::NamedTuple, protos::Tuple)
    R0 = storerowtype(intypes)
    termtypes = map((k, f) -> checktermtype(k, termtype(f, R0)), keys(terms),
        values(terms))
    return intypes,
    merge(storedbase(intypes, protos, ()), NamedTuple{keys(terms)}(termtypes))
end

function barwindowtypes(::HoistedTerms{K,hs}, intypes::NamedTuple,
    protos::Tuple) where {K,hs}
    base = storedbase(intypes, protos, hs)
    hoisted = values(NamedTuple{hs}(intypes))
    return merge(base, NamedTuple{hs}(hoisted)), merge(base, NamedTuple{K}(hoisted))
end

# A term's inferred type, which must be storable unboxed: checked here for an
# evaluated term and by `termvalues` for a hoisted one (`BarTerm`), so both are
# held to one rule over one row type.
checktermtype(k::Symbol, T) =
    concreteterm(T) ? T :
    throw(ArgumentError("barwindow row term $(repr(k)) must infer a concrete type, got $T"))

# The stored input columns, those named `hoisted` aside.
storedbase(intypes::NamedTuple, protos::Tuple, hoisted::Tuple{Vararg{Symbol}}) =
    projectinput(Base.structdiff(intypes, NamedTuple{hoisted}),
        storedcolumns(protos, Symbol[]))

# Build the state for the realized types; type-unstable, once per state and
# per widening.
function newbarwindow(protos::Tuple, outs::Val{O}, terms, n::Int,
    intypes::NamedTuple) where {O}
    checkbarterms(terms, intypes)
    rowtypes, types = barwindowtypes(terms, intypes, protos)
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
    return BarWindowState{R,storerowtype(rowtypes),O,V}(n, RowRing{R}(n + 1),
        map(fresh, tg.running), queue, tg.derived, tg.perm, protos, terms)
end

# R is the stored row type, R0 the input row type, O the output names, V
# the value type; `terms` the row terms to evaluate, or `HoistedTerms`;
# `running` holds windowed group states, `queue` the other
# monoids (`nothing` when there are none), `derived` the fieldless dependents.
# The ring holds n + 1 rows, so the row leaving is still there to downdate.
mutable struct BarWindowState{R,R0,O,V,SR<:Tuple,Q,SD<:Tuple,P,PR<:Tuple,
    TM} <: SummarizerState
    const n::Int
    const ring::RowRing{R}
    running::SR
    const queue::Q
    const derived::SD
    const perm::Val{P}
    const protos::PR   # for widenstate
    const terms::TM    # a NamedTuple of functions, or HoistedTerms
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
# input types the row terms' types were inferred from, then the row terms,
# evaluated over that whole row or (hoisted) renamed, then cut to the stored
# columns.
@inline barrow(st::BarWindowState{R,R0}, row) where {R,R0} =
    termrow(R, projectrow(R0, row), st.terms)

@inline termrow(::Type{R}, base::NamedTuple, terms::NamedTuple) where {R} =
    projectrow(R, merge(base, map(f -> f(base), terms)))
@inline termrow(::Type{R}, base::NamedTuple, ::HoistedTerms) where {R} =
    R(Tuple(base))

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
