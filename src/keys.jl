# The keyed stores' dictionary (issue #116). Base hashes a NamedTuple through
# `Tuple(nt)`, which isn't inferred once a field type is a `Union` (from four
# fields on 1.13, two on 1.10), so a key column admitting `missing` cost every
# lookup a tuple, two boxed hashes and a dispatch. Methods on Base's
# `NamedTuple` would be piracy, so a store hashes a package-owned `Key` instead,
# field by field. The wrapper lives *inside* the dictionary: callers pass and
# receive plain NamedTuple keys, so emission, ordering and error messages never
# see it.

# A stored key. `hash` and `isequal` are written out per field, so each field's
# `Union` splits and nothing builds a tuple. They are defined across NamedTuple
# types with the same names and agree field by field, so a lookup key of
# narrower or numerically different types (an `Int` against a stored
# `Float64`) still finds its entry without conversion.
struct Key{NT<:NamedTuple}
    nt::NT
end

const KEYSEED = 0x6b1f3a59d2c47e08 % UInt

@generated function Base.hash(k::Key{NT}, h::UInt) where {NT}
    ex = :(h + KEYSEED)
    for i in 1:fieldcount(NT)
        ex = :(hash(getfield(getfield(k, :nt), $i), $ex))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

@generated function Base.isequal(a::Key{NamedTuple{KN,A}},
    b::Key{NamedTuple{KN,B}}) where {KN,A,B}
    ex = :true
    for i in length(KN):-1:1
        x = :(getfield(getfield(a, :nt), $i))
        y = :(getfield(getfield(b, :nt), $i))
        ex = :(isequal($x, $y) && $ex)
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

# A `Dict{K,V}` over NamedTuple keys `K`, storing them as `Key{K}`. A lookup
# wraps its key as given (no conversion, as above); an insert converts it to
# `K`, a no-op for keys built in the store's type (`keyat`, `keyvalues`).
# Iteration yields `Pair{K,V}`, so `collect`, `sortedgroups` and the generic
# `AbstractDict` functions see plain keys.
struct KeyDict{K<:NamedTuple,V} <: AbstractDict{K,V}
    d::Dict{Key{K},V}
    KeyDict{K,V}() where {K,V} = new(Dict{Key{K},V}())
    function KeyDict{K,V}(kvs) where {K,V}
        d = Dict{Key{K},V}()
        for (k, v) in kvs
            d[Key{K}(k)] = v
        end
        return new(d)
    end
end

@inline Base.length(kd::KeyDict) = length(kd.d)
@inline Base.isempty(kd::KeyDict) = isempty(kd.d)
@inline Base.get(kd::KeyDict, k::NamedTuple, default) = get(kd.d, Key(k), default)
@inline Base.get(f::Base.Callable, kd::KeyDict, k::NamedTuple) = get(f, kd.d, Key(k))
@inline Base.get!(f::Base.Callable, kd::KeyDict{K}, k::NamedTuple) where {K} =
    get!(f, kd.d, Key{K}(k))
@inline Base.get!(kd::KeyDict{K}, k::NamedTuple, v) where {K} =
    get!(kd.d, Key{K}(k), v)
@inline Base.getindex(kd::KeyDict, k::NamedTuple) = kd.d[Key(k)]
@inline Base.setindex!(kd::KeyDict{K}, v, k::NamedTuple) where {K} =
    (kd.d[Key{K}(k)] = v; kd)
@inline Base.haskey(kd::KeyDict, k::NamedTuple) = haskey(kd.d, Key(k))
@inline Base.delete!(kd::KeyDict, k::NamedTuple) = (delete!(kd.d, Key(k)); kd)
@inline Base.empty!(kd::KeyDict) = (empty!(kd.d); kd)
@inline Base.sizehint!(kd::KeyDict, n::Integer) = (sizehint!(kd.d, n); kd)
@inline Base.values(kd::KeyDict) = values(kd.d)

@inline function Base.iterate(kd::KeyDict{K,V}, s...) where {K,V}
    y = iterate(kd.d, s...)
    y === nothing && return nothing
    p, st = y
    return (Pair{K,V}(first(p).nt, last(p)), st)
end

# In place, as Dict's own `filter!`; the generic one gathers the doomed keys
# into a fresh vector per call (the window close filters once per tick).
Base.filter!(f, kd::KeyDict{K,V}) where {K,V} =
    (filter!(p -> f(Pair{K,V}(first(p).nt, last(p))), kd.d); kd)
