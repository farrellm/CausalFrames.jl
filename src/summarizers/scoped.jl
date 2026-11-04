# A hidden dependency under private output names (issue #149). Dependents read
# their dependencies by output name from one flat NamedTuple of values, so two
# different summarizers can never share a name. When a hidden dependency's
# names clash with a different summarizer's, `prototypes` renames it and
# redirects the reads of its dependents; anything else is folded unwrapped, so
# a call without a clash compiles exactly as before.
#
# The plan `P` is `(D, A, MD, MA, TD, TA)`: the inner summarizer's declared
# output names `D`, emitted as `A`; the dependency names it reads, `MD`, taken
# from the values under `MA`; and the hoisted bar terms its window reads, `TD`,
# taken from the input columns `TA` they moved to with a renamed owner. The
# wrapper keeps the inner summarizer's structure (one type per structural
# supertype), since `tiering` picks a tier by it, and its state wraps the inner
# state, which stays a singleton when the inner one is, so a renamed dependent
# still takes the derived tier.
struct ScopedPlain{P,S<:Summarizer} <: Summarizer
    inner::S
end
struct ScopedMonoid{P,S<:Summarizer} <: MonoidSummarizer
    inner::S
end
struct ScopedGroup{P,S<:Summarizer} <: GroupSummarizer
    inner::S
end
const Scoped{P} = Union{ScopedPlain{P},ScopedMonoid{P},ScopedGroup{P}}

# `s` emitting `A` for its declared names `D`, reading `MD` from `MA` and the
# columns `TD` from `TA`.
function scoped(@nospecialize(s::Summarizer), D::Tuple, A::Tuple, MD::Tuple, MA::Tuple,
    TD::Tuple, TA::Tuple)
    P = (D, A, MD, MA, TD, TA)
    W =
        s isa GroupSummarizer ? ScopedGroup :
        s isa MonoidSummarizer ? ScopedMonoid :
        ScopedPlain
    return W{P,typeof(s)}(s)
end

scopeplan(::Scoped{P}) where {P} = P

# Delegates every state operation to `inner`. A new interface function must be
# delegated below too: test/summarizers.jl finds the interface by its
# `SummarizerState`/`DerivedState` fallbacks and checks each is delegated.
struct ScopedState{P,S<:SummarizerState} <: SummarizerState
    inner::S
end
ScopedState{P}(st::S) where {P,S<:SummarizerState} = ScopedState{P,S}(st)

emptyvalue(s::Scoped) = renamevalue(Val(scopeplan(s)), emptyvalue(s.inner))
fresh(s::Scoped, intypes::NamedTuple) =
    ScopedState{scopeplan(s)}(fresh(s.inner, aliascolumns(Val(scopeplan(s)), intypes)))
freshwindowed(s::Scoped, intypes::NamedTuple) =
    ScopedState{scopeplan(s)}(
        freshwindowed(s.inner, aliascolumns(Val(scopeplan(s)), intypes)))

fresh(st::ScopedState{P}) where {P} = ScopedState{P}(fresh(st.inner))
@inline fresh!(st::ScopedState{P}) where {P} = ScopedState{P}(fresh!(st.inner))
@inline update!(st::ScopedState{P}, row) where {P} =
    update!(st.inner, aliasrow(Val(P), row))
@inline downdate!(st::ScopedState{P}, row) where {P} =
    downdate!(st.inner, aliasrow(Val(P), row))
combine!(dest::ScopedState{P}, a::ScopedState{P}, b::ScopedState{P}) where {P} =
    combine!(dest.inner, a.inner, b.inner)
widenstate(st::ScopedState{P}, intypes::NamedTuple) where {P} =
    ScopedState{P}(widenstate(st.inner, aliascolumns(Val(P), intypes)))
isinvertible(st::ScopedState) = isinvertible(st.inner)
@inline needsrefold(st::ScopedState) = needsrefold(st.inner)
@inline value(st::ScopedState{P}) where {P} = renamevalue(Val(P), value(st.inner))
@inline value(st::ScopedState{P}, vals::NamedTuple) where {P} =
    renamevalue(Val(P), value(st.inner, aliasvalues(Val(P), vals)))

# The inner value under the emitted names: a projection onto the declared names
# (in that order, whatever order the inner value lists them in), relabelled.
# Generated over the declared field types, so a `Union{Missing,_}` field keeps
# its type: relabelling through `values` types the result by the values, which
# leaves it abstract and the emission dispatching. A value already under the
# emitted names, in order, is returned as it is.
@generated function renamevalue(::Val{P}, v::NamedTuple) where {P}
    D, A = P[1], P[2]
    D === A && fieldnames(v) === A && return :v
    T = NamedTuple{A,Tuple{(fieldtype(v, d) for d in D)...}}
    fields = (:(getfield(v, $(QuoteNode(d)))) for d in D)
    return Expr(:block, Expr(:meta, :inline), Expr(:new, T, fields...))
end

# `nt` with each field of `from` also under its name in `to`, overriding
# whatever else holds that name.
@inline aliasnames(::Val{from}, ::Val{to}, nt::NamedTuple) where {from,to} =
    isempty(to) ? nt : merge(nt, renamevalue(Val((from, to)), nt))

# The values with each renamed dependency also under the name the inner
# summarizer reads.
@inline aliasvalues(::Val{P}, vals::NamedTuple) where {P} =
    aliasnames(Val(P[4]), Val(P[3]), vals)

# The input types with each moved hoisted term also under the name the inner
# summarizer's window reads.
@inline aliascolumns(::Val{P}, intypes::NamedTuple) where {P} =
    aliasnames(Val(P[6]), Val(P[5]), intypes)

# A row read through the names the inner summarizer's window reads: rows reach
# a state as any property-access row (a column table's row, say), and a window
# reads them by `getproperty`, so a view relabelling the moved terms serves
# every row type without copying one.
@inline aliasrow(::Val{P}, row) where {P} =
    isempty(P[5]) ? row : AliasedRow{P[6],P[5]}(row)

struct AliasedRow{from,to,R}
    row::R
end
AliasedRow{from,to}(row::R) where {from,to,R} = AliasedRow{from,to,R}(row)
@inline Base.getproperty(r::AliasedRow{from,to}, c::Symbol) where {from,to} =
    getproperty(getfield(r, :row), aliasedname(Val(from), Val(to), c))
# `row[:name]`, as `update!` promises.
@inline Base.getindex(r::AliasedRow, c::Symbol) = getproperty(r, c)
# Each name once: a row name among `to` reads the aliased column instead.
Base.propertynames(r::AliasedRow{from,to}) where {from,to} =
    (Iterators.filter(∉(to), propertynames(getfield(r, :row)))..., to...)

# The column `c` is read from: `from[j]` for `c === to[j]`, else `c`. A chain of
# comparisons, so a constant `c` folds it away.
@generated function aliasedname(::Val{from}, ::Val{to}, c::Symbol) where {from,to}
    ex = :c
    for (f, t) in reverse(collect(zip(from, to)))
        ex = :(c === $(QuoteNode(t)) ? $(QuoteNode(f)) : $ex)
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end
