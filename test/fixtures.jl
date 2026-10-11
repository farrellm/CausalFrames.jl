# A test-local multi-valued summarizer: the minimum and maximum of a column.
# The column is a type parameter and the state's value type comes from the
# input schema. A monoid but not a group, for addrollingcolumns' custom-monoid
# path.
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

# A structure-hiding wrapper: delegates to the wrapped summarizer (reusing its
# state) but subtypes plain Summarizer, so it takes the window transforms'
# refold tier, the oracle the fast paths are tested against.
struct Opaque{S<:Summarizer} <: Summarizer
    inner::S
end

CausalFrames.emptyvalue(o::Opaque) = CausalFrames.emptyvalue(o.inner)
CausalFrames.fresh(o::Opaque, intypes::NamedTuple) =
    CausalFrames.fresh(o.inner, intypes)
CausalFrames.dependencies(o::Opaque) =
    map(Opaque, CausalFrames.dependencies(o.inner))

# Records the input columns its states are built from, declaring `C` as its
# input columns (`nothing`, the default, keeps every column), so a test can
# see what the window transforms' projection keeps.
struct Recording{S<:Summarizer,C} <: Summarizer
    inner::S
    seen::Vector{Any}
end
Recording(s::Summarizer, cols) = Recording{typeof(s),cols}(s, Any[])

CausalFrames.emptyvalue(r::Recording) = CausalFrames.emptyvalue(r.inner)
function CausalFrames.fresh(r::Recording, intypes::NamedTuple)
    push!(r.seen, keys(intypes))
    return CausalFrames.fresh(r.inner, intypes)
end
CausalFrames.inputcolumns(::Recording{S,C}) where {S,C} = C

# Opaque's monoid counterpart: a group hidden down to a monoid, so it takes the
# tree tier rather than the running tier, putting a built-in group's `combine!`
# under a window.
struct AsMonoid{S<:Summarizer} <: MonoidSummarizer
    inner::S
end

CausalFrames.emptyvalue(o::AsMonoid) = CausalFrames.emptyvalue(o.inner)
CausalFrames.fresh(o::AsMonoid, intypes::NamedTuple) =
    CausalFrames.fresh(o.inner, intypes)
CausalFrames.dependencies(o::AsMonoid) =
    map(AsMonoid, CausalFrames.dependencies(o.inner))

# A test-local summarizer whose output column is illegally named :time.
struct BadTime <: Summarizer end
CausalFrames.emptyvalue(::BadTime) = (; time = 0)

# A dependent summarizer whose dependencies are dependent: the population
# variance from the first two raw moments (TestVar -> Moment ->
# Count/SumPower), so expansion is transitive and both moments share a Count.
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

# A group summarizer whose state is non-invertible once its column widens to
# Float64, driving the window transforms' demotion of an accumulator from the
# running tier to the tree (no built-in does this). The output name is a type
# parameter so `value` infers.
struct FragileSum{C} <: GroupSummarizer end
FragileSum(column::Symbol) = FragileSum{column}()
mutable struct FragileSumState{C,N,T} <: SummarizerState
    total::T
end
CausalFrames.emptyvalue(::FragileSum{C}) where {C} =
    NamedTuple{(Symbol(C, :_fragile),)}((0,))
CausalFrames.fresh(::FragileSum{C}, intypes::NamedTuple) where {C} =
    FragileSumState{C,Symbol(C, :_fragile),intypes[C]}(zero(intypes[C]))
CausalFrames.fresh(::FragileSumState{C,N,T}) where {C,N,T} =
    FragileSumState{C,N,T}(zero(T))
CausalFrames.update!(st::FragileSumState{C}, row) where {C} =
    (st.total += getproperty(row, C); nothing)
CausalFrames.downdate!(st::FragileSumState{C}, row) where {C} =
    (st.total -= getproperty(row, C); nothing)
CausalFrames.combine!(dest::FragileSumState, a, b) =
    (dest.total = a.total + b.total; nothing)
CausalFrames.value(st::FragileSumState{C,N,T}) where {C,N,T} =
    NamedTuple{(N,),Tuple{T}}((st.total,))
CausalFrames.widenstate(st::FragileSumState{C,N}, intypes::NamedTuple) where {C,N} =
    FragileSumState{C,N,intypes[C]}(convert(intypes[C], st.total))
CausalFrames.isinvertible(::FragileSumState{C,N,Float64}) where {C,N} = false

# FragileSum's mirror: non-invertible while its column is Int, invertible once
# it widens to Float64, so a widening *promotes* it from the tree to the running
# tier (no built-in does this), which needs a buffer the tree-only call didn't
# keep.
struct LateSum{C} <: GroupSummarizer end
LateSum(column::Symbol) = LateSum{column}()
mutable struct LateSumState{C,N,T} <: SummarizerState
    total::T
end
CausalFrames.emptyvalue(::LateSum{C}) where {C} =
    NamedTuple{(Symbol(C, :_late),)}((0,))
CausalFrames.fresh(::LateSum{C}, intypes::NamedTuple) where {C} =
    LateSumState{C,Symbol(C, :_late),intypes[C]}(zero(intypes[C]))
CausalFrames.fresh(::LateSumState{C,N,T}) where {C,N,T} = LateSumState{C,N,T}(zero(T))
CausalFrames.update!(st::LateSumState{C}, row) where {C} =
    (st.total += getproperty(row, C); nothing)
CausalFrames.downdate!(st::LateSumState{C}, row) where {C} =
    (st.total -= getproperty(row, C); nothing)
CausalFrames.combine!(dest::LateSumState, a, b) =
    (dest.total = a.total + b.total; nothing)
CausalFrames.value(st::LateSumState{C,N,T}) where {C,N,T} =
    NamedTuple{(N,),Tuple{T}}((st.total,))
CausalFrames.widenstate(st::LateSumState{C,N}, intypes::NamedTuple) where {C,N} =
    LateSumState{C,N,intypes[C]}(convert(intypes[C], st.total))
CausalFrames.isinvertible(::LateSumState{C,N,Int}) where {C,N} = false

# A dependent spanning every window tier: its dependencies are a group (Sum,
# running), a monoid (Product, tree) and a plain summarizer (Opaque(Sum) under
# its own name, refold), and its fieldless state reads all three at emission.
struct TierSpan{C} <: Summarizer end
TierSpan(column::Symbol) = TierSpan{column}()

struct TierSpanState{C,N,S,P,Q} <: SummarizerState end

# The refold dependency: Sum under another name, hidden from the structure.
struct PlainSum{C} <: Summarizer end
PlainSum(column::Symbol) = PlainSum{column}()
CausalFrames.emptyvalue(::PlainSum{C}) where {C} =
    NamedTuple{(Symbol(C, :_plainsum),)}((0,))
CausalFrames.fresh(::PlainSum{C}, intypes::NamedTuple) where {C} =
    CausalFrames.accumfresh(CausalFrames.ColumnTerm{C}(), Symbol(C, :_plainsum),
        CausalFrames.sumtype(intypes[C]))

CausalFrames.dependencies(::TierSpan{C}) where {C} =
    (Sum(C), Product(C), PlainSum(C))
CausalFrames.emptyvalue(::TierSpan{C}) where {C} =
    NamedTuple{(Symbol(C, :_span),)}((missing,))
CausalFrames.fresh(::TierSpan{C}, ::NamedTuple) where {C} =
    TierSpanState{C,Symbol(C, :_span),Symbol(C, :_sum),Symbol(C, :_product),
        Symbol(C, :_plainsum)}()
CausalFrames.fresh(st::TierSpanState) = st
CausalFrames.update!(::TierSpanState, row) = nothing
function CausalFrames.value(::TierSpanState{C,N,S,P,Q},
    vals::NamedTuple) where {C,N,S,P,Q}
    V = Base.promote_op((s, p, q) -> s + p - 2q, fieldtype(typeof(vals), S),
        fieldtype(typeof(vals), P), fieldtype(typeof(vals), Q))
    return NamedTuple{(N,),Tuple{V}}((vals[S] + vals[P] - 2 * vals[Q],))
end

# A test-local summarizer that depends on itself, for cycle detection.
struct Loopy <: Summarizer end
CausalFrames.dependencies(::Loopy) = (Loopy(),)
CausalFrames.emptyvalue(::Loopy) = (; loopy = missing)

# A deterministic linear-congruential sequence of n values in 0:(m - 1), so
# property tests need no RNG dependency and failures reproduce exactly.
function lcgsequence(seed::Integer, n::Int, m::Int)
    vals = Vector{Int}(undef, n)
    s = UInt64(seed)
    for i in 1:n
        s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        vals[i] = Int((s >>> 33) % UInt64(m))
    end
    return vals
end

# A test-local summarizer whose one dependency declares a row term, as a
# downstream package's hidden term would.
rangeterm(r) = r.high - r.low
struct RangeTotal <: Summarizer end
struct RangeTotalState <: SummarizerState end
CausalFrames.emptyvalue(::RangeTotal) = (; rangetotal = 0.0)
CausalFrames.dependencies(::RangeTotal) = (Sum(:range => rangeterm),)
CausalFrames.fresh(::RangeTotal, ::NamedTuple) = RangeTotalState()
CausalFrames.fresh(st::RangeTotalState) = st
CausalFrames.update!(::RangeTotalState, row) = nothing
CausalFrames.value(::RangeTotalState, vals::NamedTuple) =
    (; rangetotal = vals.range_sum)

# A summarizer embedding a count window mid-state, as a recursive indicator
# would: the mean of the last `n` rows, read back per row. The window's state
# type is a parameter, as `barwindow`'s docstring asks.
struct LastNMean{C} <: Summarizer
    n::Int
end
LastNMean(column::Symbol, n::Int) = LastNMean{column}(n)
struct LastNMeanState{C,N,W<:SummarizerState} <: SummarizerState
    window::W
end
LastNMeanState{C,N}(window::W) where {C,N,W} = LastNMeanState{C,N,W}(window)
CausalFrames.emptyvalue(::LastNMean{C}) where {C} =
    NamedTuple{(Symbol(C, :_lastnmean),)}((missing,))
CausalFrames.fresh(s::LastNMean{C}, intypes::NamedTuple) where {C} =
    LastNMeanState{C,Symbol(C, :_lastnmean)}(
        CausalFrames.barwindow(Mean(C), s.n, NamedTuple{(C,)}((intypes[C],))))
CausalFrames.fresh(st::LastNMeanState{C,N}) where {C,N} =
    LastNMeanState{C,N}(CausalFrames.fresh(st.window))
CausalFrames.update!(st::LastNMeanState, row) = CausalFrames.update!(st.window, row)
# The declared field type, not the value's: `missing` until the window fills.
function CausalFrames.value(st::LastNMeanState{C,N}) where {C,N}
    v = CausalFrames.value(st.window)
    return NamedTuple{(N,),Tuple{fieldtype(typeof(v), 1)}}((v[1],))
end

# How many of this process's file descriptors refer to `path`: Linux only, via
# /proc, so callers guard on `Sys.islinux()`. For checking that a failed run
# leaves no file open.
function openfds(path::AbstractString)
    target = realpath(path)
    n = 0
    for fd in readdir("/proc/self/fd"; join = true)
        link = try
            readlink(fd)
        catch
            continue    # closed between the listing and the read
        end
        n += link == target
    end
    return n
end

# A source emitting one chunk and then failing: the run errors after a sink
# downstream has already received data.
failingsource() = CausalPipeline() do _
    (
        i == 1 ? DataFrame(time = [1, 2], x = [1.0, 2.0]) :
        throw(ArgumentError("failingsource: boom")) for i in 1:2
    )
end

# Run `p` expecting `failingsource`'s error, then check that the sink writing
# `path` released it.
function checkreleased(p::CausalPipeline, path::AbstractString)
    @test_throws ArgumentError scan(Context(0, 10), p)
    Sys.islinux() && @test openfds(path) == 0
    return nothing
end
