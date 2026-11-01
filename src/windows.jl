# The window-summarization transform. A clock supplies the ticks; at each tick
# τ the rows with time in [τ - lookback, τ) are summarized and emitted at τ.
# The data stream drives a chunkmap and pulls the clock on demand, as in
# intervalize; the window bookkeeping is addrollingcolumns' (a buffer of
# admitted rows with an eviction head), sampled at ticks rather than rows.
# Keyless output is a grid (one row per tick), the dense output over the empty
# key; keyed output is sparse, with one empty row marking a key whose window
# has emptied.
#
# The tiers are rolling.jl's (tiers.jl): running groups (O(1) amortized per
# row), two-stack queues expired and queried at each tick (O(1) amortized per
# row), and refolds (O(window) per tick). Type-unstable setup runs once per
# chunk; the kernels take concretely typed arguments.

"""
    summarizewindows(clock, lookback, summarizers; key = nothing,
                     keyset = nothing) -> (CausalPipeline -> CausalPipeline)
    summarizewindows(p::CausalPipeline, clock, lookback, summarizers;
                     ...) -> CausalPipeline

A transform summarizing a trailing window at every clock tick: at each tick
`τ`, the rows with time in `[τ - lookback, τ)` are summarized and emitted at
`τ`, with columns `time`, the key columns, then the summaries; the input
columns are dropped. The input runs over `[start - lookback, stop)`, so the
first tick sees a full window. Unlike [`intervalize`](@ref)'s intervals,
windows may overlap or leave gaps.

# Arguments
- `clock`: a pipeline whose `:time` column gives the ticks (other columns are
  ignored), such as [`clock`](@ref).
- `lookback`: the window length; non-negative and subtractable from the time
  type.
- `summarizers`: a [`Summarizer`](@ref) or a collection of them.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key every tick emits one row, an empty window its
  summarizers' empty values. With a key each tick emits one row per key with
  rows in its window, sorted by key, plus one row of empty values for each key
  whose window has just emptied, so a downstream [`asofjoin`](@ref) sees it go
  empty.
- `keyset = nothing`: the key values, declared up front (requires `key`), as for
  [`intervalize`](@ref). Every tick then emits one row per declared key, in
  declared order.

Each summarizer's window slides in O(1) per row when it is a
[`GroupSummarizer`](@ref), in O(1) amortized per row when it is another
[`MonoidSummarizer`](@ref), and is re-folded in O(window) otherwise.
"""
function summarizewindows(clk::CausalPipeline, lookback, summarizers;
    key = nothing, keyset = nothing)
    lookback isa Bars && throw(
        ArgumentError(
            "summarizewindows lookback must be a time span, got $lookback; " *
            "`Bars` look-backs are for `addrollingcolumns`",
        ),
    )
    (; keycols, protos, terms, ks, keynames, outs) =
        summaryspec(summarizers, key, "summarizewindows"; keyset, grid = true)
    cfg = WindowConfig(keycols, keynames, lookback, protos, outs, ks, terms,
        storedcolumns(protos, keycols))
    contexts = ctx -> (widenstart(ctx, lookback, "summarizewindows lookback"), ctx)
    # Each tick's window lies within the widened run, but a keyed tick's
    # vanish rows are decided against the tick before it, which a run from a
    # later start never saw.
    startfree = ks !== nothing   # keyless runs as the declared `nokeyset`
    return function (@nospecialize(p::CausalPipeline),)
        return pipenode((p, clk); contexts, startfree) do ctx, ins
            T = timetype(ctx)
            st = WindowState{T}(IntervalCursor{T}(openinput(ins, 2)), cfg)
            # Through `st`, whose `cfg` is erased, so the iterator's type and
            # the per-chunk driver are the same for every summarizer set.
            return chunkmap(c -> windowchunk!(st, st.cfg, c), openinput(ins, 1);
                flush = () -> windowflush!(st, st.cfg))
        end
    end
end
summarizewindows(p::CausalPipeline, clk::CausalPipeline, lookback, summarizers;
    kwargs...) = summarizewindows(clk, lookback, summarizers; kwargs...)(p)

# The declared key set (`Nothing` when sparse, `nokeyset` when keyless) is a
# type parameter, so it costs no per-row branch.
struct WindowConfig{KN,LB,P<:Tuple,O,KS<:Union{Nothing,KeySet},TM<:NamedTuple}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    lookback::LB
    protos::P
    outs::Val{O}
    ks::KS
    terms::TM  # the row terms, from `prototypes`
    stored::Union{Nothing,Tuple{Vararg{Symbol}}}  # from `storedcolumns`
end

# Per-run state. The untyped fields are per-chunk setup; per-row work sits
# behind the kernels' function barrier.
mutable struct WindowState{T}
    const cfg::WindowConfig  # abstract, a dispatch per chunk
    cur::IntervalCursor{T}
    ticks::Vector{T}    # pulled ticks not yet closed
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    tiers::Any          # WindowTiers for the realized state types
    valtype::Any        # the emitted value type
    head::Int           # first buffered row not yet evicted
    prevkeys::Any       # Vector{K}: keys emitted with rows at the previous tick
    checked::Bool       # key columns validated against the input
    WindowState{T}(cur, cfg::WindowConfig) where {T} = new{T}(cfg, cur, T[],
        nothing, nothing, nothing, 1, nothing, false)
end

# The window structures for one tiering (tiers.jl), R being the stored row type.
# The buffer holds the admitted rows not yet evicted, in time order, for the
# running tier's eviction and the refold tier's folds; the monoid queues own
# their rows, so a monoid-only call keeps no buffer. In every tier a key is
# present exactly when its window has rows (the running table deletes a group
# with its last row, an emptied key's queue is dropped, the refold table is
# filled at the tick from the live rows), so the first tier present, the
# primary, gives the keys to emit.
struct WindowTiers{R,B,G,MQ,GT,TG<:Tiering,SC}
    buffer::B        # Vector{R}, or nothing
    running::G       # RunningTable{K,SR}, or nothing
    monoids::MQ      # KeyDict{K,MonoidQueues{ST,R,T}}, or nothing
    refold::GT       # GroupTable{K,SF}, or nothing
    tg::TG           # the partition of the states into tiers
    scratch::SC      # Vector{Pair{K,V}} over the primary's dict: the sort buffer
end

WindowTiers{R}(buffer::B, running::G, monoids::MQ, refold::GT, tg::TG,
    scratch::SC) where {R,B,G,MQ,GT,TG,SC} =
    WindowTiers{R,B,G,MQ,GT,TG,SC}(buffer, running, monoids, refold, tg, scratch)

@inline primarygroups(t::WindowTiers) =
    primarygroups(t.running, t.monoids, t.refold)
@inline primarygroups(rt::RunningTable, monoids, refold) = rt.groups
@inline primarygroups(::Nothing, monoids::AbstractDict, refold) = monoids
@inline primarygroups(::Nothing, ::Nothing, gt::GroupTable) = gt.table

# Build the tiers for the first realized schema, or rebuild them from the live
# rows for a widened one (`rebuildbuffer`, `rebuildmonoids`), which is correct
# for every transition, including one that demotes an accumulator from the
# running tier to the monoid tier. The refold table is filled only at ticks, so it
# starts empty.
function preparewindows!(st::WindowState, @nospecialize(cfg::WindowConfig),
    types::NamedTuple)
    tg = tiering(cfg.protos, types)
    K = storekeytype(types, cfg.keynames)
    R = storerowtype(types)
    st.prevkeys =
        st.prevkeys === nothing ? K[] : convert(Vector{K}, st.prevkeys)
    old = st.tiers
    # The live rows start at the old head, if the old tiers buffered rows.
    oldhead = old === nothing || old.buffer === nothing ? 1 : st.head
    buffer = rebuildbuffer(tg, R, old)
    running =
        isempty(tg.running) ? nothing :
        replayrunning!(RunningTable{K,typeof(tg.running)}(), buffer, oldhead,
            tg.running, cfg.keynames)
    monoids = rebuildmonoids(tg, K, R, types.time, old, oldhead, cfg.keynames,
        (cfg.lookback,))
    refold = isempty(tg.refold) ? nothing : GroupTable{K,typeof(tg.refold)}()
    st.head = buffer === nothing ? 1 : oldhead
    groups = primarygroups(running, monoids, refold)
    st.tiers = WindowTiers{R}(buffer, running, monoids, refold, tg,
        Pair{K,valtype(groups)}[])
    st.valtype = tieredvaluetype(tg, cfg.protos, cfg.outs)
    return nothing
end

# The emitted row type and the empty row. The key type is the data's when
# sparse, the declared one when dense.
windowkeytype(::Nothing, st::WindowState) = eltype(st.prevkeys)
windowkeytype(::KeySet{K}, st::WindowState) where {K} = K

windowtypes(st::WindowState{T}, @nospecialize(cfg::WindowConfig)) where {T} =
    gridtypes(T, windowkeytype(cfg.ks, st), st.valtype, cfg.protos, cfg.outs)

function windowchunk!(st::WindowState{T}, @nospecialize(cfg::WindowConfig),
    c::DataFrame) where {T}
    if !st.checked
        checkkeycolumns(cfg.keycols, c, "summarizewindows")
        st.checked = true
    end
    # Every tick is closed, so no later row falls in any window.
    exhausted(st.cur) && isempty(st.ticks) && return nothing
    nt = projectinput(terminput(c, cfg.terms, "summarizewindows"), cfg.stored)
    types = promotetypes(st.types, chunktypes(nt))
    moved = types != st.types
    st.types = types
    moved && preparewindows!(st, cfg, types)
    pullpast!(st.ticks, st.cur, last(nt.time))
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head, closed = windowrows!(rows, st.tiers, st.head, nt, st.ticks,
        st.prevkeys, cfg.lookback, cfg.keynames, cfg.outs, emptyrow, cfg.ks)
    deleteat!(st.ticks, 1:closed)
    exhausted(st.cur) && isempty(st.ticks) && releasewindows!(st)
    return rowsframe(rows)
end

# Dead rows before the head are dropped once they dominate the buffer:
# amortized O(1) per row. Called at each tick, so the buffer stays near the
# window's size.
compacthead!(::Nothing, head::Int) = head
compacthead!(buffer::Vector, head::Int) = head - compactdead!(buffer, head - 1)

# After the last tick nothing more is emitted, so the live rows (the buffer,
# groups and queues) are released rather than held to the end of the stream.
function releasewindows!(st::WindowState)
    t = st.tiers
    t.buffer === nothing || empty!(t.buffer)
    st.head = 1
    t.running === nothing || empty!(t.running.groups)
    t.monoids === nothing || empty!(t.monoids)
    return nothing
end

function windowflush!(st::WindowState{T}, @nospecialize(cfg::WindowConfig)) where {T}
    pullpast!(st.ticks, st.cur, nothing)
    isempty(st.ticks) && return nothing
    if st.tiers === nothing
        # No data ever arrived: empty rows when dense, nothing when sparse.
        cfg.ks === nothing && return nothing
        return emptygrid(st.ticks, cfg.ks.keys, emptyvalues(cfg.protos, cfg.outs))
    end
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head = flushwindows!(rows, st.tiers, st.head, st.ticks, st.prevkeys,
        cfg.lookback, cfg.keynames, cfg.outs, emptyrow, cfg.ks)
    empty!(st.ticks)
    return rowsframe(rows)
end

# --- kernels ---------------------------------------------------------------
#
# Called with concretely typed arguments; an absent tier compiles away. For
# each row at time s, every pending tick τ <= s is closed first (windows are
# half-open, so the row is in none of them), then the row is admitted. Returns
# the eviction head and the number of ticks closed.
function windowrows!(rows::Vector{RT}, tiers::WindowTiers{R}, head::Int,
    nt::NamedTuple, ticks::Vector{T}, prevkeys::Vector, lookback,
    keynames::Val, outs::Val, emptyrow,
    ks::Union{Nothing,KeySet}) where {RT,R,T}
    admitks = admissionkeys(ks)
    bi = 1
    nb = length(ticks)
    for i in eachindex(nt.time)
        s = @inbounds nt.time[i]
        while bi <= nb && @inbounds(ticks[bi]) <= s
            head = closewindow!(rows, @inbounds(ticks[bi]), tiers, head,
                prevkeys, lookback, keynames, outs, emptyrow, ks)
            bi += 1
        end
        admitwindow!(tiers, rowat(R, nt, i), keynames, admitks, (lookback,))
    end
    return head, bi - 1
end

function flushwindows!(rows::Vector, tiers::WindowTiers, head::Int,
    ticks::Vector, prevkeys::Vector, lookback, keynames::Val, outs::Val,
    emptyrow, ks::Union{Nothing,KeySet})
    for τ in ticks
        head = closewindow!(rows, τ, tiers, head, prevkeys, lookback, keynames,
            outs, emptyrow, ks)
    end
    return head
end

# Admit one row into every tier. A row may already be outside the next tick's
# window (a look-back shorter than the tick spacing); that tick's eviction
# downdates it right back out. A declared keyset (`admissionkeys`) is checked
# when the primary tier makes a key's group or queue, or on every admission
# when the primary is the refold table, which is filled only at ticks.
@inline function admitwindow!(tiers::WindowTiers, row, keynames::Val,
    ks::Union{Nothing,KeySet}, lookbacks::Tuple)
    pushrow!(tiers.buffer, row)
    k = keyvalues(row, keynames)
    admitwinrunning!(tiers.running, tiers.tg.running, k, row, ks, keynames)
    admitwinmonoid!(tiers.monoids, tiers.tg.monoid, k, row,
        tiers.running === nothing ? ks : nothing, keynames, lookbacks)
    tiers.running === nothing && tiers.monoids === nothing &&
        checkdeclared(ks, row, keynames)
    return nothing
end

@inline admitwinrunning!(::Nothing, stateprotos, k, row, ks, keynames::Val) =
    nothing
@inline admitwinrunning!(rt::RunningTable, stateprotos::Tuple, k, row, ks,
    keynames::Val) = admitgroup!(rt, stateprotos, k, row, ks, keynames)

@inline admitwinmonoid!(::Nothing, stateprotos, k, row, ks, keynames::Val,
    lookbacks) = nothing
@inline function admitwinmonoid!(monoids::KeyDict{K,Q}, stateprotos::Tuple, k,
    row, ks, keynames::Val, lookbacks::Tuple) where {K,Q<:MonoidQueues}
    mq = get!(monoids, k) do
        checkdeclared(ks, row, keynames)
        newqueues(Q, stateprotos, length(lookbacks))
    end
    queuepush!(mq, row, lookbacks)
    return nothing
end

# Close tick τ: evict the rows that have left its window, downdating them out of
# the running tier; expire each key's queue to the window start, dropping
# emptied ones; fold the live rows into the refold table; emit; then retire the
# refold states to the pool.
@noinline function closewindow!(rows::Vector, τ, tiers::WindowTiers, head::Int,
    prevkeys::Vector, lookback, keynames::Val, outs::Val, emptyrow,
    ks::Union{Nothing,KeySet})
    head = evicthead!(tiers.buffer, head, τ, lookback, tiers.running, keynames)
    head = compacthead!(tiers.buffer, head)
    closemonoids!(tiers.monoids, τ, lookback)
    foldrefold!(tiers.refold, tiers.buffer, head, tiers.tg.refold, keynames)
    emitgroups!(rows, τ, primarygroups(tiers), tiers.scratch, prevkeys, tiers,
        outs, emptyrow, ks)
    retirerefold!(tiers.refold)
    return head
end

# Every admitted row is before τ, so a queue's window is its rows from `lo`. An
# emptied key's queue is dropped, since ticks only advance.
@inline closemonoids!(::Nothing, τ, lookback) = nothing
@inline function closemonoids!(monoids::AbstractDict, τ, lookback)
    filter!(monoids) do (_, mq)
        expirequeue!(mq, 1, τ, lookback)
        return hasrows(mq, 1)
    end
    return nothing
end

@inline foldrefold!(::Nothing, buffer, head::Int, stateprotos, keynames::Val) =
    nothing
@inline function foldrefold!(gt::GroupTable{K,S}, buffer::Vector, head::Int,
    stateprotos::S, keynames::Val) where {K,S}
    for j in head:length(buffer)
        row = @inbounds buffer[j]
        updateall!(groupstates!(gt, keyvalues(row, keynames), stateprotos), row)
    end
    return nothing
end

@inline retirerefold!(::Nothing) = nothing
@inline retirerefold!(gt::GroupTable) = retireall!(gt)

# One tier's states for key k, whose entry in the primary tier is g: `()` for
# an absent tier, g itself for the primary, a lookup by key for the others,
# and for a queue its borrowed states (`livestates`, since a present key's
# window holds rows). The running tier, when present, is always the primary.
@inline tierstates(::Nothing, k, g) = ()
@inline tierstates(::RunningTable{K,S}, k, g::RunningGroup{S}) where {K,S<:Tuple} =
    g.states
@inline tierstates(::KeyDict{K,V}, k, mq::V) where {K,V<:MonoidQueues} =
    livestates(mq, 1)
@inline tierstates(monoids::KeyDict{K,V}, k, g) where {K,V<:MonoidQueues} =
    tierstates(monoids, k, monoids[k])
@inline tierstates(::GroupTable{K,S}, k, states::S) where {K,S<:Tuple} = states
@inline tierstates(gt::GroupTable, k, g) = gt.table[k]

@inline tiervalues(tiers::WindowTiers, k, g, outs::Val) = summaryvalues(
    mergestates(tiers.tg.perm,
        (tierstates(tiers.running, k, g), tierstates(tiers.monoids, k, g),
            tierstates(tiers.refold, k, g), tiers.tg.derived)), outs)

# Emit one tick from the primary tier's groups: without a declared keyset, the
# present keys sorted into the reused scratch, then `emitwindow!`; with one
# (keyless included), `emitdense!`.
@inline function emitgroups!(rows::Vector, τ, groups::AbstractDict,
    scratch::Vector{<:Pair}, prevkeys::Vector, tiers::WindowTiers, outs::Val,
    emptyrow, ::Nothing)
    sortinto!(scratch, groups)
    return emitwindow!(rows, τ, scratch, prevkeys, tiers, outs, emptyrow)
end
@inline emitgroups!(rows::Vector, τ, groups::AbstractDict, scratch,
    prevkeys::Vector, tiers::WindowTiers, outs::Val, emptyrow, ks::KeySet) =
    emitdense!(rows, τ, groups, ks, tiers, outs, emptyrow)

# One row per declared key, in declared order: the key's summary when it has a
# group, else the empty values. The declared key is looked up as given: a
# `KeyDict` matches it field by field, whatever its field types.
function emitdense!(rows::Vector{RT}, τ, groups::AbstractDict, ks::KeySet,
    tiers::WindowTiers, outs::Val, emptyrow) where {RT}
    # `haskey` then indexing hashes each key twice, but `get(groups, k,
    # nothing)` would box a refold primary's entry, a state tuple, in a Union
    # with Nothing.
    for k in ks.keys
        if haskey(groups, k)
            push!(rows,
                convert(RT, gridrow(τ, k, tiervalues(tiers, k, groups[k], outs))))
        else
            push!(rows, convert(RT, gridrow(τ, k, emptyrow)))
        end
    end
    return rows
end

# Emit one sparse tick: the present keys in key order, merged with an empty
# row for each key present at the previous tick and absent now; the present
# keys then become `prevkeys`. Both lists are sorted by `keyisless`, so one
# merge walk suffices.
function emitwindow!(rows::Vector{RT}, τ, present::Vector{<:Pair},
    prevkeys::Vector{K}, tiers::WindowTiers, outs::Val, emptyrow) where {RT,K}
    j = 1
    np = length(prevkeys)
    for (k, g) in present
        while j <= np && keyisless(@inbounds(prevkeys[j]), k)
            push!(rows, convert(RT, gridrow(τ, @inbounds(prevkeys[j]), emptyrow)))
            j += 1
        end
        j <= np && isequal(@inbounds(prevkeys[j]), k) && (j += 1)
        push!(rows, convert(RT, gridrow(τ, k, tiervalues(tiers, k, g, outs))))
    end
    while j <= np
        push!(rows, convert(RT, gridrow(τ, @inbounds(prevkeys[j]), emptyrow)))
        j += 1
    end
    empty!(prevkeys)
    for (k, _) in present
        push!(prevkeys, k)
    end
    return rows
end
