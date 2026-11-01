# The keyed stores' dictionary (issue #116): a `Dict` over package-owned `Key`
# wrappers hashed and compared field by field, since Base's NamedTuple `hash`
# isn't inferred over `Union` fields. Rationale: DESIGN.md and src/CLAUDE.md.

# A stored key. `hash` and `isequal` agree field by field across NamedTuple
# types with the same names, so a lookup key of other field types (an `Int`
# against a stored `Float64`) finds its entry without conversion.
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
    field(k, i) = :(getfield(getfield($k, :nt), $i))
    fieldeq(i) = :(isequal($(field(:a, i)), $(field(:b, i))))
    n = length(KN)
    ex = n == 0 ? :true : fieldeq(n)
    for i in (n-1):-1:1
        ex = :($(fieldeq(i)) && $ex)
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

# Key equality outside a store (the refold and window-close scans), field by
# field as a `KeyDict` lookup compares, not through Base's `Tuple(nt)`.
@inline keyisequal(a::NamedTuple, b::NamedTuple) = isequal(Key(a), Key(b))

# A `Dict{K,V}` over NamedTuple keys `K`, storing them as `Key{K}`. A lookup
# wraps its key as given (no conversion, as above); an insert converts it to
# `K`, a no-op for keys built in the store's type (`keyat`, `keyvalues`).
# Iteration yields `Pair{K,V}`, so `collect`, `sortedgroups` and the generic
# `AbstractDict` functions see plain keys.
struct KeyDict{K<:NamedTuple,V} <: AbstractDict{K,V}
    d::Dict{Key{K},V}
    KeyDict{K,V}() where {K,V} = new(Dict{Key{K},V}())
    KeyDict{K,V}(d::Dict{Key{K},V}) where {K,V} = new(d)
    function KeyDict{K,V}(kvs) where {K,V}
        d = Dict{Key{K},V}()
        Base.IteratorSize(kvs) isa Union{Base.HasLength,Base.HasShape} &&
            sizehint!(d, length(kvs))
        for (k, v) in kvs
            d[Key{K}(k)] = v
        end
        return new(d)
    end
end
KeyDict{K,V}(ps::Pair...) where {K,V} = KeyDict{K,V}(ps)

@inline Base.length(kd::KeyDict) = length(kd.d)
@inline Base.get(kd::KeyDict, k::NamedTuple, default) = get(kd.d, Key(k), default)
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
# So `empty` keeps a store a KeyDict, not the generic plain `Dict`, and `copy`
# copies the table rather than rehashing every entry.
Base.copy(kd::KeyDict{K,V}) where {K,V} = KeyDict{K,V}(copy(kd.d))
Base.empty(::KeyDict, ::Type{K}, ::Type{V}) where {K<:NamedTuple,V} = KeyDict{K,V}()

# A stored entry as the plain pair callers see.
@inline unkey(p::Pair{Key{K},V}) where {K,V} = Pair{K,V}(first(p).nt, last(p))

@inline function Base.iterate(kd::KeyDict, s...)
    y = iterate(kd.d, s...)
    y === nothing && return nothing
    return (unkey(first(y)), last(y))
end

# In place, as Dict's own `filter!`; the generic one gathers the doomed keys
# into a fresh vector per call (the window close filters once per tick).
Base.filter!(f, kd::KeyDict) = (filter!(p -> f(unkey(p)), kd.d); kd)
