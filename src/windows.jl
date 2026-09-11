# The window-summarization transform. A clock pipeline supplies the tick times;
# at each tick τ the rows with time in [τ - lookback, τ) are summarized and
# emitted at τ. The data stream drives a chunkmap and the clock is pulled on
# demand, exactly as in intervalize; the window bookkeeping is addrollingcolumns'
# — a buffer of admitted rows with an eviction head — sampled at the ticks
# rather than at every row. Keyless output is a grid (one row per tick); keyed
# output is sparse, with one empty row marking a key whose window has emptied.
#
# Two window algorithms, chosen from the summarizers' structure like
# rolling.jl's: all GroupSummarizers slide per-key running states (update! on
# admission, downdate! on eviction, O(1) amortized per row); anything else
# re-folds each window at its tick, O(window) per tick. A widening that defeats
# downdate! (isinvertible) demotes running to re-fold, which needs nothing but
# the buffer both modes keep. Per the summarize.jl conventions, the
# type-unstable setup happens once per chunk and the kernels take concretely
# typed arguments behind function barriers.

"""
    summarizewindows(clock, lookback, summarizers;
                     key = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizewindows(p::CausalPipeline, clock, lookback, summarizers;
                     key = nothing) -> CausalPipeline

A transform summarizing a trailing window at every tick of a **clock** pipeline:
at each time `τ` in the clock's `:time` column, the rows with time in
`[τ - lookback, τ)` — inclusive of the window's start, exclusive of the tick
itself — are summarized and emitted at `τ`, dropping the input columns.
`summarizers` is a [`Summarizer`](@ref) or a collection of them; the output
columns are `time`, the key columns, then each summarizer's value columns. Only
the clock's `:time` column is used.

It generalizes [`intervalize`](@ref): intervalize's intervals are the gaps
between consecutive ticks, while these windows have a fixed length that may be
shorter or longer than the gaps, so windows may overlap. A `lookback` equal to a
regular clock's spacing reproduces intervalize's intervals. The input runs over
the context widened to `[start - lookback, stop)`, so the first tick already
sees a full window; this requires the time type to support subtraction (numbers
and `Dates` types do), and `lookback` must be non-negative.

Without `key` the output is a **regular grid**: every tick emits exactly one
row, an empty window included, with the summarizers' identity/missing values
(`count = 0`, `mean = missing`), so element types widen to admit them. With
`key` (a column name or collection of column names) the output is **sparse**:
each tick emits one row per key with rows in its window, sorted by key — plus
one row of empty values for each key that had rows in its window at the
previous tick and has none now. That row is what lets a consumer tracking the
latest row per key (such as [`asofjoin`](@ref)) see a key's summary go empty
rather than keep its stale value; the key is then not emitted again until its
window holds rows.

Windows over [`GroupSummarizer`](@ref)s slide in O(1) per row, subtracting rows
as they leave; anything else is re-folded over each window, O(window) per tick.

The curried form composes with `|>`; the uncurried form applies directly, so
`summarizewindows(p, clock, lookback, ss; key)` is equivalent to
`p |> summarizewindows(clock, lookback, ss; key)`.
"""
function summarizewindows(clk::CausalPipeline, lookback, summarizers;
    key = nothing)
    keycols = tokeycolumns(key)
    allunique(keycols) ||
        throw(ArgumentError("summarizewindows key columns must be unique"))
    :time in keycols && throw(
        ArgumentError(
            "time is the window dimension and may not be a summarizewindows key"),
    )
    protos, requested = prototypes(tosummarizers(summarizers), keycols)
    cfg = WindowConfig(keycols, Val(Tuple(keycols)), lookback, protos,
        Val(requested), Val(isempty(keycols)),
        candidatemode(protos) isa RunningMode ? RunningMode() : RefoldMode())
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            T = timetype(ctx)
            st = WindowState{T}(IntervalCursor{T}(clk.run(ctx)))
            return chunkmap(c -> windowstep!(st, cfg, c),
                p.run(windowcontext(ctx, lookback));
                flush = () -> windowflush!(st, cfg))
        end
    end
end
summarizewindows(p::CausalPipeline, clk::CausalPipeline, lookback, summarizers;
    kwargs...) = summarizewindows(clk, lookback, summarizers; kwargs...)(p)

# Non-negativity falls out of `start - lookback <= start`, the rollingcontext
# precedent, so the look-back type never has to be compared with zero.
function windowcontext(ctx::Context, lookback)
    start = ctx.start - lookback
    start <= ctx.start || throw(
        ArgumentError(
            "summarizewindows lookback must be non-negative, got $lookback"),
    )
    return Context(start, ctx.stop)
end

# `grid` (keyless) and the candidate mode ride in type parameters, so the
# kernels specialize on them and neither costs a per-row branch.
struct WindowConfig{KN,LB,P<:Tuple,O,G,M<:RollMode}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    lookback::LB
    protos::P
    outs::Val{O}
    grid::Val{G}
    candidate::M
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the kernels' function barrier.
mutable struct WindowState{T}
    cur::IntervalCursor{T}
    ticks::Vector{T}    # pulled ticks not yet closed
    doneticks::Bool     # the clock is exhausted
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    stateprotos::Any    # state tuple template
    buffer::Any         # Vector{R}: admitted rows, time-ordered
    head::Int           # first buffered row not yet evicted
    mode::RollMode      # effective algorithm for the realized state types
    groups::Any         # running: Dict{K,RunningGroup{S}}; re-fold: GroupTable{K,S}
    scratch::Any        # running: Vector{Pair{K,RunningGroup{S}}}, the sort buffer
    prevkeys::Any       # Vector{K}: keys emitted with rows at the previous tick
    checked::Bool       # key columns validated against the input
    WindowState{T}(cur) where {T} = new{T}(cur, T[], false, nothing, nothing,
        nothing, 1, RefoldMode(), nothing, nothing, nothing, false)
end

# Pull ticks until the last one is strictly past `tmax`, so every tick a row of
# the chunk (all `<= tmax`) can close is known, or the clock is exhausted.
# Type-unstable (the clock pull), run once per chunk.
function fillticks!(st::WindowState{T}, tmax::T) where {T}
    while !st.doneticks && (isempty(st.ticks) || @inbounds(st.ticks[end]) <= tmax)
        b = nextboundary!(st.cur)
        b === nothing ? (st.doneticks = true) : push!(st.ticks, b)
    end
    return nothing
end

function drainticks!(st::WindowState)
    while !st.doneticks
        b = nextboundary!(st.cur)
        b === nothing ? (st.doneticks = true) : push!(st.ticks, b)
    end
    return nothing
end

# The running mode needs every state invertible for the realized types; the
# tree mode rolling.jl would use for monoid-only sets is not implemented here,
# so anything short of that re-folds.
windowmode(::RefoldMode, ::Tuple) = RefoldMode()
windowmode(::RunningMode, stateprotos::Tuple) =
    all(isinvertible, stateprotos) ? RunningMode() : RefoldMode()

# Build the states, buffer and mode structures for the first realized schema,
# or rebuild them for a widened one. The running groups are replayed from the
# live rows (rolling.jl's `replaygroups!`), which is correct for every
# transition, demotion to re-fold included.
function preparewindows!(st::WindowState, cfg::WindowConfig, types::NamedTuple)
    if st.stateprotos === nothing
        st.stateprotos = newstates(cfg.protos, types)
        st.buffer = storerowtype(types)[]
        st.head = 1
    else
        st.stateprotos = widenstates(st.stateprotos, types)
        st.buffer = convert(Vector{storerowtype(types)}, st.buffer)
    end
    K = storekeytype(types, cfg.keynames)
    st.prevkeys =
        st.prevkeys === nothing ? K[] : convert(Vector{K}, st.prevkeys)
    S = typeof(st.stateprotos)
    st.mode = windowmode(cfg.candidate, st.stateprotos)
    if st.mode isa RunningMode
        st.groups = replaygroups!(Dict{K,RunningGroup{S}}(), st.buffer,
            st.head, st.stateprotos, cfg.keynames)
        st.scratch = Pair{K,RunningGroup{S}}[]
    else
        st.groups = GroupTable{K,S}()
        st.scratch = nothing
    end
    return nothing
end

# The emitted row type and the empty row, from the realized states: the summary
# values promoted field-wise with the empty values (both are emitted), behind
# `:time` and the key.
_windowrow(t, k, v) = merge((; time = t), k, v)
windowrowtype(::Type{T}, ::Type{K}, ::Type{V}) where {T,K,V} =
    Base.promote_op(_windowrow, T, K, V)

function windowtypes(st::WindowState{T}, cfg::WindowConfig) where {T}
    V = promotedvaluetype(typeof(st.stateprotos), cfg.protos, cfg.outs)
    RT = windowrowtype(T, eltype(st.prevkeys), V)
    return RT, convert(V, emptyvalues(cfg.protos, cfg.outs))
end

function windowstep!(st::WindowState{T}, cfg::WindowConfig,
    c::DataFrame) where {T}
    if !st.checked
        for k in cfg.keycols
            String(k) in names(c) || throw(
                ArgumentError(
                    "summarizewindows key column $k not found in the input"),
            )
        end
        st.checked = true
    end
    types = promotetypes(st.types, chunktypes(c))
    moved = st.types === nothing || types != st.types
    st.types = types
    moved && preparewindows!(st, cfg, types)
    nt = Tables.columntable(c)
    fillticks!(st, last(nt.time))
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head, closed =
        st.mode isa RunningMode ?
        windowrunning!(rows, st.buffer, st.head, nt, st.ticks, st.groups,
            st.stateprotos, st.scratch, st.prevkeys, cfg.lookback,
            cfg.keynames, cfg.outs, emptyrow, cfg.grid) :
        windowrefold!(rows, st.buffer, st.head, nt, st.ticks, st.groups,
            st.stateprotos, st.prevkeys, cfg.lookback, cfg.keynames, cfg.outs,
            emptyrow, cfg.grid)
    deleteat!(st.ticks, 1:closed)
    st.head = compact!(st.buffer, st.head)
    return isempty(rows) ? nothing : DataFrame(rows)
end

function windowflush!(st::WindowState{T}, cfg::WindowConfig) where {T}
    drainticks!(st)
    isempty(st.ticks) && return nothing
    if st.stateprotos === nothing
        # No data ever arrived, so no states were built: a keyless grid of empty
        # rows typed from the configs alone; nothing at all when keyed.
        cfg.grid isa Val{true} || return nothing
        e = emptyvalues(cfg.protos, cfg.outs)
        RT = windowrowtype(T, typeof((;)), typeof(e))
        return DataFrame(RT[convert(RT, _windowrow(t, (;), e)) for t in st.ticks])
    end
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head =
        st.mode isa RunningMode ?
        flushrunning!(rows, st.buffer, st.head, st.ticks, st.groups, st.scratch,
            st.prevkeys, cfg.lookback, cfg.keynames, cfg.outs, emptyrow,
            cfg.grid) :
        flushrefold!(rows, st.buffer, st.head, st.ticks, st.groups,
            st.stateprotos, st.prevkeys, cfg.lookback, cfg.keynames, cfg.outs,
            emptyrow, cfg.grid)
    empty!(st.ticks)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# --- kernels ---------------------------------------------------------------
#
# Called with concretely typed arguments. For each row at time s, every
# pending tick τ <= s is closed first — the window is half-open, so the row is
# in no window of a tick at or before it — and then the row is admitted. Both
# return (head, closed): the eviction head and how many ticks were closed.

function windowrunning!(rows::Vector{RT}, buffer::Vector{R}, head::Int,
    nt::NamedTuple, ticks::Vector{T}, groups::Dict{K,RunningGroup{S}},
    stateprotos::S, scratch::Vector{Pair{K,RunningGroup{S}}},
    prevkeys::Vector{K}, lookback, keynames::Val, outs::Val, emptyrow,
    grid::Val) where {RT,R,T,K,S<:Tuple}
    bi = 1
    nb = length(ticks)
    for i in eachindex(nt.time)
        s = @inbounds nt.time[i]
        while bi <= nb && @inbounds(ticks[bi]) <= s
            head = closerunning!(rows, @inbounds(ticks[bi]), buffer, head,
                groups, scratch, prevkeys, lookback, keynames, outs, emptyrow,
                grid)
            bi += 1
        end
        row = rowat(R, nt, i)
        push!(buffer, row)
        # A row may already be outside the next tick's window (a look-back
        # shorter than the tick spacing); the eviction at that tick downdates it
        # right back out, keeping the groups equal to buffer[head:end] per key.
        g = get!(() -> RunningGroup(map(fresh, stateprotos), 0), groups,
            keyvalues(row, keynames))
        updateall!(g.states, row)
        g.live += 1
    end
    return head, bi - 1
end

function flushrunning!(rows::Vector{RT}, buffer::Vector, head::Int,
    ticks::Vector, groups::Dict{K,RunningGroup{S}},
    scratch::Vector{Pair{K,RunningGroup{S}}}, prevkeys::Vector{K}, lookback,
    keynames::Val, outs::Val, emptyrow, grid::Val) where {RT,K,S<:Tuple}
    for τ in ticks
        head = closerunning!(rows, τ, buffer, head, groups, scratch, prevkeys,
            lookback, keynames, outs, emptyrow, grid)
    end
    return head
end

# Evict the rows that have left τ's window, deleting a group with its last row
# so that presence in the dict means rows in the window, then emit.
@inline function closerunning!(rows::Vector, τ, buffer::Vector, head::Int,
    groups::Dict{K,RunningGroup{S}}, scratch::Vector{Pair{K,RunningGroup{S}}},
    prevkeys::Vector{K}, lookback, keynames::Val, outs::Val, emptyrow,
    grid::Val) where {K,S<:Tuple}
    while head <= length(buffer) && τ - @inbounds(buffer[head]).time > lookback
        row = @inbounds buffer[head]
        k = keyvalues(row, keynames)
        g = groups[k]
        downdateall!(g.states, row)
        g.live -= 1
        g.live == 0 && delete!(groups, k)
        head += 1
    end
    empty!(scratch)
    append!(scratch, groups)
    sort!(scratch; by = groupkey)
    emitwindow!(rows, τ, scratch, prevkeys, outs, emptyrow, grid)
    return head
end

function windowrefold!(rows::Vector{RT}, buffer::Vector{R}, head::Int,
    nt::NamedTuple, ticks::Vector{T}, gt::GroupTable{K,S}, stateprotos::S,
    prevkeys::Vector{K}, lookback, keynames::Val, outs::Val, emptyrow,
    grid::Val) where {RT,R,T,K,S<:Tuple}
    bi = 1
    nb = length(ticks)
    for i in eachindex(nt.time)
        s = @inbounds nt.time[i]
        while bi <= nb && @inbounds(ticks[bi]) <= s
            head = closerefold!(rows, @inbounds(ticks[bi]), buffer, head, gt,
                stateprotos, prevkeys, lookback, keynames, outs, emptyrow, grid)
            bi += 1
        end
        push!(buffer, rowat(R, nt, i))
    end
    return head, bi - 1
end

function flushrefold!(rows::Vector{RT}, buffer::Vector, head::Int,
    ticks::Vector, gt::GroupTable{K,S}, stateprotos::S, prevkeys::Vector{K},
    lookback, keynames::Val, outs::Val, emptyrow,
    grid::Val) where {RT,K,S<:Tuple}
    for τ in ticks
        head = closerefold!(rows, τ, buffer, head, gt, stateprotos, prevkeys,
            lookback, keynames, outs, emptyrow, grid)
    end
    return head
end

# Advance the head past the rows that have left τ's window, fold the rest into
# per-key states drawn from the table's pool, emit, and retire the states back
# to the pool — `summaryvalues` has copied their values out by then, which is
# the `closecycle!` protocol.
@inline function closerefold!(rows::Vector, τ, buffer::Vector, head::Int,
    gt::GroupTable{K,S}, stateprotos::S, prevkeys::Vector{K}, lookback,
    keynames::Val, outs::Val, emptyrow, grid::Val) where {K,S<:Tuple}
    while head <= length(buffer) && τ - @inbounds(buffer[head]).time > lookback
        head += 1
    end
    for j in head:length(buffer)
        row = @inbounds buffer[j]
        updateall!(groupstates!(gt, keyvalues(row, keynames), stateprotos), row)
    end
    scratch = gt.scratch
    empty!(scratch)
    append!(scratch, gt.table)
    sort!(scratch; by = groupkey)
    emitwindow!(rows, τ, scratch, prevkeys, outs, emptyrow, grid)
    for (_, states) in scratch
        push!(gt.pool, states)
    end
    empty!(gt.table)
    return head
end

@inline windowstates(g::RunningGroup) = g.states
@inline windowstates(states::Tuple) = states

# Emit one tick. Keyless (grid): exactly one row, the summary or the empty
# values. Keyed: the present keys in key order, merged with an empty row for
# every key present at the previous tick and absent now; the present keys then
# become the previous tick's. Both lists are sorted by the tuple of key values,
# the order `groupkey` gives, so one merge walk suffices.
function emitwindow!(rows::Vector{RT}, τ, present::Vector{<:Pair},
    prevkeys::Vector{K}, outs::Val, emptyrow, ::Val{G}) where {RT,K,G}
    if G
        push!(rows,
            convert(RT,
                isempty(present) ? _windowrow(τ, (;), emptyrow) :
                _windowrow(τ, (;),
                    summaryvalues(windowstates(last(first(present))), outs))))
        return rows
    end
    j = 1
    np = length(prevkeys)
    for (k, g) in present
        while j <= np && isless(values(@inbounds(prevkeys[j])), values(k))
            push!(rows, convert(RT, _windowrow(τ, @inbounds(prevkeys[j]), emptyrow)))
            j += 1
        end
        j <= np && isequal(@inbounds(prevkeys[j]), k) && (j += 1)
        push!(rows,
            convert(RT, _windowrow(τ, k, summaryvalues(windowstates(g), outs))))
    end
    while j <= np
        push!(rows, convert(RT, _windowrow(τ, @inbounds(prevkeys[j]), emptyrow)))
        j += 1
    end
    empty!(prevkeys)
    for (k, _) in present
        push!(prevkeys, k)
    end
    return rows
end
