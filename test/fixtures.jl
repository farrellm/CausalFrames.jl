# A test-local multi-valued summarizer exercising the NamedTuple interface:
# tracks the minimum and maximum of a column. Also the worked example of the
# config/state split — the column rides in a type parameter, and the state's
# value type comes from the input schema.
struct MinMax{C} <: Summarizer end
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

# A test-local summarizer whose output column is illegally named :time.
struct BadTime <: Summarizer end
CausalFrames.emptyvalue(::BadTime) = (; time = 0)
