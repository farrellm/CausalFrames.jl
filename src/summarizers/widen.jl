# Folding Float16 and Float32 at Float64, and narrowing back.
#
# At their own precision a square or a product passes floatmax past a deviation
# of 256 (Float16) or 1.8e19 (Float32), a fourth power past 16 or 4.3e9, and a
# value's `n·Σd²` past a few hundred Float16 rows, though the statistic itself
# is in range. So the accumulators behind the second-order statistics
# (`CoMoment`, `CentralMoments`, `CrossProduct`) fold these types at Float64
# (`widenfold`), and every dependent narrows only its own value back to its
# inputs' float type, so output types are unchanged.
#
# The accumulator says what it was built for: its value is a `Widened{P}`,
# with P the (promoted, non-missing) input type it folds, so a dependent reads
# the type to narrow to from the type of its dependency values (`narrowof`)
# and needs no type parameter or `widenstate` of its own (issue #139). A
# plain dependency value (a `Sum`, or a test's literal) stands for its own type.
widenfold(::Type{T}) where {T} = T
widenfold(::Type{Float16}) = Float64
widenfold(::Type{Float32}) = Float64

# `widenfold` of a possibly `Missing`-admitting type, `missing` kept.
widenfoldmissing(::Type{Missing}) = Missing
widenfoldmissing(::Type{T}) where {T} =
    maybemissing(widenfold(nonmissingtype(T)), Missing <: T)

# A widened accumulator's value V, folded from input of type P. It never
# reaches an output: only dependents read it, through `unwiden`.
struct Widened{P,V}
    value::V
end
Widened{P}(v::V) where {P,V} = Widened{P,V}(v)

@inline unwiden(w::Widened) = w.value
@inline unwiden(x) = x

# The value type behind a dependency's declared field type F, `missing` kept.
widenedvaluetype(::Type{Missing}) = Missing
widenedvaluetype(::Type{F}) where {F} =
    maybemissing(widenedinner(nonmissingtype(F)), Missing <: F)
widenedinner(::Type{Widened{P,V}}) where {P,V} = V
widenedinner(::Type{T}) where {T} = T

# The input type behind it: P for a widened value, a plain value's own type,
# and nothing (`Union{}`, promotion's identity) for an all-missing one.
widenedsource(::Type{Missing}) = Union{}
widenedsource(::Type{F}) where {F} = widenedsourceinner(nonmissingtype(F))
widenedsourceinner(::Type{Widened{P,V}}) where {P,V} = P
widenedsourceinner(::Type{T}) where {T} = T

# The type a dependent narrows its value to, from the types of its inputs:
# their promoted float type where that was widened, else `Nothing`, for no
# narrowing.
function narrowtype(Ts::Type...)
    T = promote_type(map(nonmissingtype, Ts)...)
    return widenfold(T) === T ? Nothing : T
end

# `narrowtype` over the sources of the fields NS of a dependency-value
# NamedTuple type V. Generated, so it is a constant on every emission path.
@generated narrowof(::Type{V}, ::Val{NS}) where {V<:NamedTuple,NS} =
    narrowtype(map(n -> widenedsource(fieldtype(V, n)), NS)...)

# The narrowed value and its type, `missing` kept.
narrowval(::Type{Nothing}, x) = x
narrowval(::Type{E}, x) where {E} = x === missing ? missing : convert(E, x)
narrowedtype(::Type{Nothing}, ::Type{V}) where {V} = V
narrowedtype(::Type{E}, ::Type{V}) where {E,V} =
    V === Missing ? Missing : maybemissing(E, Missing <: V)

# Compared and hashed by value, as the plain values were, so the folds'
# equivalence tests (a refold against a slide, a combine against a fold)
# compare numbers, -0.0 == 0.0 included, not bits.
Base.:(==)(a::Widened{P}, b::Widened{P}) where {P} = a.value == b.value
Base.isequal(a::Widened{P}, b::Widened{P}) where {P} = isequal(a.value, b.value)
Base.hash(w::Widened, h::UInt) = hash(w.value, hash(:Widened, h))
