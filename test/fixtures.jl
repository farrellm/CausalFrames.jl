# A test-local multi-valued summarizer exercising the NamedTuple interface:
# tracks the minimum and maximum of a column. Also the worked example of the
# config/state split — the column rides in a type parameter, and the state's
# value type comes from the input schema. A monoid (combine! below) but not a
# group, exercising the custom-monoid path of addrollingcolumns.
struct MinMax{C} <: MonoidSummarizer end
MinMax(column::Symbol) = MinMax{column}()

mutable struct MinMaxState{C,L,H,T} <: SummarizerState
    seen::Bool
    lo::T
    hi::T
    MinMaxState{C,L,H,T}() where {C,L,H,T} = new{C,L,H,T}(false)
end

CausalFrames.emptyvalue(::MinMax{C}) where {C} =
    NamedTuple{(Symbol(C, :_min), Symbol(C, :_max))}((missing, missing))
CausalFrames.fresh(::MinMax{C}, intypes::NamedTuple) where {C} =
    MinMaxState{C,Symbol(C, :_min),Symbol(C, :_max),intypes[C]}()
CausalFrames.fresh(::MinMaxState{C,L,H,T}) where {C,L,H,T} = MinMaxState{C,L,H,T}()
function CausalFrames.update!(st::MinMaxState{C}, row) where {C}
    v = getproperty(row, C)
    st.lo = st.seen ? min(st.lo, v) : v
    st.hi = st.seen ? max(st.hi, v) : v
    st.seen = true
    return nothing
end
CausalFrames.value(st::MinMaxState{C,L,H,T}) where {C,L,H,T} =
    NamedTuple{(L, H),Tuple{T,T}}((st.lo, st.hi))
function CausalFrames.combine!(dest::MinMaxState{C,L,H,T},
    a::MinMaxState{C,L,H,T},
    b::MinMaxState{C,L,H,T}) where {C,L,H,T}
    if a.seen && b.seen
        lo = min(a.lo, b.lo)
        hi = max(a.hi, b.hi)
        dest.lo = lo
        dest.hi = hi
        dest.seen = true
    elseif a.seen
        dest.lo = a.lo
        dest.hi = a.hi
        dest.seen = true
    elseif b.seen
        dest.lo = b.lo
        dest.hi = b.hi
        dest.seen = true
    else
        dest.seen = false
    end
    return nothing
end

# A structure-hiding wrapper: delegates the interface to the wrapped
# summarizer (reusing its state unchanged) but subtypes plain Summarizer, so
# any tuple containing it takes addrollingcolumns's re-fold fallback — the
# oracle the fast paths are differentially tested against.
struct Opaque{S<:Summarizer} <: Summarizer
    inner::S
end

CausalFrames.emptyvalue(o::Opaque) = CausalFrames.emptyvalue(o.inner)
CausalFrames.fresh(o::Opaque, intypes::NamedTuple) =
    CausalFrames.fresh(o.inner, intypes)
CausalFrames.dependencies(o::Opaque) =
    map(Opaque, CausalFrames.dependencies(o.inner))

# A test-local summarizer whose output column is illegally named :time.
struct BadTime <: Summarizer end
CausalFrames.emptyvalue(::BadTime) = (; time = 0)

# A test-local dependent summarizer whose dependencies are themselves
# dependent: the population variance from the first two raw moments
# (TestVar -> Moment -> Count/SumPower), so expansion is transitive and the
# Count underneath both moments is shared.
struct TestVar{C} <: Summarizer end
TestVar(column::Symbol) = TestVar{column}()

struct TestVarState{C,N,M1,M2} <: SummarizerState end

CausalFrames.dependencies(::TestVar{C}) where {C} = (Moment(C, 1), Moment(C, 2))
CausalFrames.emptyvalue(::TestVar{C}) where {C} =
    NamedTuple{(Symbol(C, :_var),)}((missing,))
CausalFrames.fresh(::TestVar{C}, ::NamedTuple) where {C} =
    TestVarState{C,Symbol(C, :_var),Symbol(C, :_moment_1),
        Symbol(C, :_moment_2)}()
CausalFrames.fresh(st::TestVarState) = st
CausalFrames.update!(::TestVarState, row) = nothing
function CausalFrames.value(::TestVarState{C,N,M1,M2},
    vals::NamedTuple) where {C,N,M1,M2}
    V = Base.promote_op((m2, m1) -> m2 - m1^2, fieldtype(typeof(vals), M2),
        fieldtype(typeof(vals), M1))
    return NamedTuple{(N,),Tuple{V}}((vals[M2] - vals[M1]^2,))
end

# A test-local summarizer that depends on itself, for cycle detection.
struct Loopy <: Summarizer end
CausalFrames.dependencies(::Loopy) = (Loopy(),)
CausalFrames.emptyvalue(::Loopy) = (; loopy = missing)

# A tiny deterministic linear-congruential sequence for property tests, so
# the suite needs no RNG dependency and failures reproduce exactly: n values
# drawn from 0:(m - 1).
function lcgsequence(seed::Integer, n::Int, m::Int)
    vals = Vector{Int}(undef, n)
    s = UInt64(seed)
    for i in 1:n
        s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        vals[i] = Int((s >>> 33) % UInt64(m))
    end
    return vals
end
