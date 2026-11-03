# A hidden dependency under private output names (issue #149). Dependents read
# their dependencies by output name from one flat NamedTuple of values, so two
# different summarizers can never share a name. When a hidden dependency's
# names clash with a different summarizer's, `prototypes` renames it and
# redirects the reads of its dependents; anything else is folded unwrapped, so
# a call without a clash compiles exactly as before.
#
# The plan `P` is `(D, A, MD, MA)`: the inner summarizer's declared output names
# `D`, emitted as `A`, and the dependency names it reads, `MD`, taken from the
# values under `MA`. The wrapper keeps the inner summarizer's structure (one
# type per structural supertype), since `tiering` picks a tier by it, and its
# state wraps the inner state, which stays a singleton when the inner one is,
# so a renamed dependent still takes the derived tier.
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

# `s` emitting `A` for its declared names `D`, reading `MD` from `MA`.
function scoped(@nospecialize(s::Summarizer), D::Tuple, A::Tuple, MD::Tuple, MA::Tuple)
    P = (D, A, MD, MA)
    W =
        s isa GroupSummarizer ? ScopedGroup :
        s isa MonoidSummarizer ? ScopedMonoid :
        ScopedPlain
    return W{P,typeof(s)}(s)
end

scopeplan(::Scoped{P}) where {P} = P

struct ScopedState{P,S<:SummarizerState} <: SummarizerState
    inner::S
end
scopestate(::Val{P}, st::SummarizerState) where {P} = ScopedState{P,typeof(st)}(st)

emptyvalue(s::Scoped) = renamevalue(Val(scopeplan(s)), emptyvalue(s.inner))
fresh(s::Scoped, intypes::NamedTuple) =
    scopestate(Val(scopeplan(s)), fresh(s.inner, intypes))
freshwindowed(s::Scoped, intypes::NamedTuple) =
    scopestate(Val(scopeplan(s)), freshwindowed(s.inner, intypes))

fresh(st::ScopedState{P}) where {P} = scopestate(Val(P), fresh(st.inner))
@inline fresh!(st::ScopedState{P}) where {P} = scopestate(Val(P), fresh!(st.inner))
@inline update!(st::ScopedState, row) = update!(st.inner, row)
@inline downdate!(st::ScopedState, row) = downdate!(st.inner, row)
combine!(dest::ScopedState{P}, a::ScopedState{P}, b::ScopedState{P}) where {P} =
    combine!(dest.inner, a.inner, b.inner)
widenstate(st::ScopedState{P}, intypes::NamedTuple) where {P} =
    scopestate(Val(P), widenstate(st.inner, intypes))
isinvertible(st::ScopedState) = isinvertible(st.inner)
@inline needsrefold(st::ScopedState) = needsrefold(st.inner)
@inline value(st::ScopedState{P}) where {P} = renamevalue(Val(P), value(st.inner))
@inline value(st::ScopedState{P}, vals::NamedTuple) where {P} =
    renamevalue(Val(P), value(st.inner, aliasvalues(Val(P), vals)))

# The inner value under the emitted names. Generated, so the renaming is
# constant field accesses whatever order the inner value lists its names in.
@generated function renamevalue(::Val{P}, v::NamedTuple) where {P}
    D, A = P[1], P[2]
    fields = [:(getfield(v, $(QuoteNode(d)))) for d in D]
    return Expr(:block, Expr(:meta, :inline),
        :(NamedTuple{$A}(($(fields...),))))
end

# The values with each renamed dependency also under the name the inner
# summarizer reads, overriding whatever else holds that name.
@generated function aliasvalues(::Val{P}, vals::NamedTuple) where {P}
    MD, MA = P[3], P[4]
    isempty(MD) && return :vals
    fields = [:(getfield(vals, $(QuoteNode(a)))) for a in MA]
    return Expr(:block, Expr(:meta, :inline),
        :(merge(vals, NamedTuple{$MD}(($(fields...),)))))
end
