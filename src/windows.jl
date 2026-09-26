# The window-summarization transform. A clock pipeline supplies the tick times;
# at each tick τ the rows with time in [τ - lookback, τ) are summarized and
# emitted at τ. The data stream drives a chunkmap and the clock is pulled on
# demand, exactly as in intervalize; the window bookkeeping is addrollingcolumns'
# — a buffer of admitted rows with an eviction head — sampled at the ticks
# rather than at every row. Keyless output is a grid (one row per tick); keyed
# output is sparse, with one empty row marking a key whose window has emptied.
#
# Each summarizer's window algorithm follows its own structure, as in
# rolling.jl (tiers.jl): groups slide per-key running states (update! on
# admission, downdate! on eviction, O(1) amortized per row), other monoids
# append rows to per-key segment trees and fold each window from O(log window)
# partial combinations at its tick, and anything else re-folds each window at
# its tick, O(window) per tick. The tree differs from rolling's in when it
# recombines: rolling queries after every row, so each append updates its
# ancestors, while here nothing reads a tree between ticks, so a tick's rows are
# appended as bare leaves and their ancestors recombined together at the tick
# (`treesync!`) — about one combine per row instead of log(window). Per the
# summarize.jl conventions, the type-unstable setup happens once per chunk and
# the kernels take concretely typed arguments behind function barriers.

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
[`GroupSummarizer`](@ref), in O(log window) per key per tick when it is another
[`MonoidSummarizer`](@ref), and is re-folded in O(window) otherwise.
"""
function summarizewindows(clk::CausalPipeline, lookback, summarizers;
    key = nothing, keyset = nothing)
    keycols = keycolumns(key, "summarizewindows")
    protos, requested = prototypes(tosummarizers(summarizers), keycols, "summarizewindows")
    ks = tokeyset(keyset, keycols, "summarizewindows")
    cfg = WindowConfig(keycols, Val(Tuple(keycols)), lookback, protos,
        Val(requested), Val(isempty(keycols)), ks)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            T = timetype(ctx)
            st = WindowState{T}(IntervalCursor{T}(clk.run(ctx)))
            return chunkmap(c -> windowstep!(st, cfg, c),
                p.run(widenstart(ctx, lookback, "summarizewindows lookback"));
                flush = () -> windowflush!(st, cfg))
        end
    end
end
summarizewindows(p::CausalPipeline, clk::CausalPipeline, lookback, summarizers;
    kwargs...) = summarizewindows(clk, lookback, summarizers; kwargs...)(p)

# `grid` (keyless) and the declared key set (`Nothing` unless dense) ride in
# type parameters, so the kernels specialize on them and neither costs a per-row
# branch.
struct WindowConfig{KN,LB,P<:Tuple,O,G,KS<:Union{Nothing,KeySet}}
    keycols::Vector{Symbol}
    keynames::Val{KN}
    lookback::LB
    protos::P
    outs::Val{O}
    grid::Val{G}
    ks::KS
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the kernels' function barrier.
mutable struct WindowState{T}
    cur::IntervalCursor{T}
    ticks::Vector{T}    # pulled ticks not yet closed
    types::Union{Nothing,NamedTuple}  # promotion of every input schema seen
    tiers::Any          # WindowTiers for the realized state types
    valtype::Any        # the emitted value type
    head::Int           # first buffered row not yet evicted
    prevkeys::Any       # Vector{K}: keys emitted with rows at the previous tick
    checked::Bool       # key columns validated against the input
    WindowState{T}(cur) where {T} = new{T}(cur, T[], nothing, nothing,
        nothing, 1, nothing, false)
end

# The window structures for one tiering (tiers.jl), R being the stored row type.
# The buffer holds the admitted rows not yet evicted, in time order, for the
# running tier's eviction and the refold tier's folds; the trees own their rows,
# so a tree-only call keeps no buffer. In each tier present, presence of a key
# means rows in its window: the running table deletes a group with its last row,
# a tree whose window empties is dropped, and the refold table is filled at the
# tick from the live rows. So the first tier present — the primary — gives the
# keys to emit, and the others are looked up by key.
struct WindowTiers{R,B,G,TR,GT,SR<:Tuple,ST<:Tuple,SF<:Tuple,SD<:Tuple,P,SC}
    buffer::B        # Vector{R}, or nothing
    running::G       # RunningTable{K,SR}, or nothing
    trees::TR        # Dict{K,SegTree{ST,R,T}}, or nothing
    refold::GT       # GroupTable{K,SF}, or nothing
    runprotos::SR
    treeprotos::ST
    refoldprotos::SF
    derived::SD
    perm::Val{P}
    scratch::SC      # Vector{Pair{K,V}} over the primary's dict: the sort buffer
end

WindowTiers{R}(buffer::B, running::G, trees::TR, refold::GT, runprotos::SR,
    treeprotos::ST, refoldprotos::SF, derived::SD, perm::Val{P},
    scratch::SC) where {R,B,G,TR,GT,SR,ST,SF,SD,P,SC} =
    WindowTiers{R,B,G,TR,GT,SR,ST,SF,SD,P,SC}(buffer, running, trees, refold,
        runprotos, treeprotos, refoldprotos, derived, perm, scratch)

@inline primarygroups(t::WindowTiers) = primarygroups(t.running, t.trees, t.refold)
@inline primarygroups(rt::RunningTable, trees, refold) = rt.groups
@inline primarygroups(::Nothing, trees::AbstractDict, refold) = trees
@inline primarygroups(::Nothing, ::Nothing, gt::GroupTable) = gt.table

# Build the tiers for the first realized schema, or rebuild them for a widened
# one, always from the live rows — correct for every transition, including an
# accumulator the widening demotes from the running tier to the tree. New trees
# replay the buffer, surviving ones the old trees' own rows; a buffer the old
# tiers did not keep is gathered from the old trees (`treebuffer`). The refold
# table is filled only at ticks, so it starts empty.
function preparewindows!(st::WindowState, cfg::WindowConfig, types::NamedTuple)
    tg = tiering(cfg.protos, types)
    K = storekeytype(types, cfg.keynames)
    R = storerowtype(types)
    st.prevkeys =
        st.prevkeys === nothing ? K[] : convert(Vector{K}, st.prevkeys)
    old = st.tiers
    buffer = rebuildbuffer(tg, R, old)
    (old === nothing || old.buffer === nothing) && (st.head = 1)
    running =
        isempty(tg.running) ? nothing :
        replayrunning!(RunningTable{K,typeof(tg.running)}(), buffer, st.head,
            tg.running, cfg.keynames)
    trees = rebuildtrees(tg, K, R, types.time, old, st.head, cfg.keynames)
    refold = isempty(tg.refold) ? nothing : GroupTable{K,typeof(tg.refold)}()
    buffer === nothing && (st.head = 1)
    groups = primarygroups(running, trees, refold)
    st.tiers = WindowTiers{R}(buffer, running, trees, refold, tg.running,
        tg.tree, tg.refold, tg.derived, tg.perm,
        Pair{K,valtype(groups)}[])
    st.valtype = tieredvaluetype(tg, cfg.protos, cfg.outs)
    return nothing
end

# The emitted row type and the empty row, from the realized states: the summary
# values promoted field-wise with the empty values (both are emitted), behind
# `:time` and the key — the data's key type when sparse, the declared one when
# dense.
windowkeytype(::Nothing, st::WindowState) = eltype(st.prevkeys)
windowkeytype(::KeySet{K}, st::WindowState) where {K} = K

function windowtypes(st::WindowState{T}, cfg::WindowConfig) where {T}
    V = st.valtype
    RT = gridrowtype(T, windowkeytype(cfg.ks, st), V)
    return RT, convert(V, emptyvalues(cfg.protos, cfg.outs))
end

function windowstep!(st::WindowState{T}, cfg::WindowConfig,
    c::DataFrame) where {T}
    if !st.checked
        checkkeycolumns(cfg.keycols, c, "summarizewindows")
        st.checked = true
    end
    # The clock is exhausted and every tick closed, so no row from here on can
    # fall in any window: drop the chunk rather than admit rows nothing evicts.
    exhausted(st.cur) && isempty(st.ticks) && return nothing
    types = promotetypes(st.types, chunktypes(c))
    moved = st.types === nothing || types != st.types
    st.types = types
    moved && preparewindows!(st, cfg, types)
    nt = Tables.columntable(c)
    pullpast!(st.ticks, st.cur, last(nt.time))
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head, closed = windowrows!(rows, st.tiers, st.head, nt, st.ticks,
        st.prevkeys, cfg.lookback, cfg.keynames, cfg.outs, emptyrow, cfg.grid,
        cfg.ks)
    deleteat!(st.ticks, 1:closed)
    exhausted(st.cur) && isempty(st.ticks) && releasewindows!(st)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# Dead rows accumulate at the front of the buffer as the head advances;
# dropping them only when they dominate keeps the cost amortized O(1) per
# admitted row. Called at each tick's eviction, so the buffer stays near the
# window's size rather than growing through a chunk.
compacthead!(::Nothing, head::Int) = head
function compacthead!(buffer::Vector, head::Int)
    dead = head - 1
    if dead >= 64 && 2 * dead >= length(buffer)
        deleteat!(buffer, 1:dead)
        return 1
    end
    return head
end

# Past the clock's last tick nothing is emitted again, so the live rows — the
# buffer, and the trees, which own theirs — are released rather than held to the
# end of the stream, as intervalize drops its trailing rows.
function releasewindows!(st::WindowState)
    t = st.tiers
    t.buffer === nothing || empty!(t.buffer)
    st.head = 1
    t.running === nothing || empty!(t.running.groups)
    t.trees === nothing || empty!(t.trees)
    return nothing
end

# The no-data grid, typed from the configs alone: one empty row per tick
# keyless, one per tick per declared key when dense.
function emptywindowgrid(ticks::Vector{T}, ::Nothing, e) where {T}
    RT = gridrowtype(T, typeof((;)), typeof(e))
    return DataFrame(RT[convert(RT, gridrow(t, (;), e)) for t in ticks])
end
function emptywindowgrid(ticks::Vector{T}, ks::KeySet{K}, e) where {T,K}
    RT = gridrowtype(T, K, typeof(e))
    rows = RT[convert(RT, gridrow(t, k, e)) for t in ticks for k in ks.keys]
    return isempty(rows) ? nothing : DataFrame(rows)
end

function windowflush!(st::WindowState{T}, cfg::WindowConfig) where {T}
    pullpast!(st.ticks, st.cur, nothing)
    isempty(st.ticks) && return nothing
    if st.tiers === nothing
        # No data ever arrived, so no states were built: the grid of empty rows
        # when keyless or dense; nothing at all when sparse.
        (cfg.grid isa Val{true} || cfg.ks !== nothing) || return nothing
        return emptywindowgrid(st.ticks, cfg.ks,
            emptyvalues(cfg.protos, cfg.outs))
    end
    RT, emptyrow = windowtypes(st, cfg)
    rows = RT[]
    st.head = flushwindows!(rows, st.tiers, st.head, st.ticks, st.prevkeys,
        cfg.lookback, cfg.keynames, cfg.outs, emptyrow, cfg.grid, cfg.ks)
    empty!(st.ticks)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# --- kernels ---------------------------------------------------------------
#
# Called with concretely typed arguments; an absent tier compiles to nothing.
# For each row at time s, every pending tick τ <= s is closed first — the
# window is half-open, so the row is in no window of a tick at or before it —
# and then the row is admitted. Returns (head, closed): the eviction head and
# how many ticks were closed.
function windowrows!(rows::Vector{RT}, tiers::WindowTiers{R}, head::Int,
    nt::NamedTuple, ticks::Vector{T}, prevkeys::Vector, lookback,
    keynames::Val, outs::Val, emptyrow, grid::Val,
    ks::Union{Nothing,KeySet}) where {RT,R,T}
    bi = 1
    nb = length(ticks)
    for i in eachindex(nt.time)
        s = @inbounds nt.time[i]
        while bi <= nb && @inbounds(ticks[bi]) <= s
            head = closewindow!(rows, @inbounds(ticks[bi]), tiers, head,
                prevkeys, lookback, keynames, outs, emptyrow, grid, ks)
            bi += 1
        end
        admitwindow!(tiers, rowat(R, nt, i), keynames, ks)
    end
    return head, bi - 1
end

function flushwindows!(rows::Vector, tiers::WindowTiers, head::Int,
    ticks::Vector, prevkeys::Vector, lookback, keynames::Val, outs::Val,
    emptyrow, grid::Val, ks::Union{Nothing,KeySet})
    for τ in ticks
        head = closewindow!(rows, τ, tiers, head, prevkeys, lookback, keynames,
            outs, emptyrow, grid, ks)
    end
    return head
end

# Admit one row into every tier. A row may already be outside the next tick's
# window (a look-back shorter than the tick spacing); the eviction at that tick
# downdates it right back out. A declared keyset is checked where the primary
# tier makes a key's group or tree — a row whose key already has one was
# declared — and on admission when the primary is the refold table, which is
# filled only at ticks. The tree appends a bare leaf, synced at the tick.
@inline function admitwindow!(tiers::WindowTiers, row, keynames::Val,
    ks::Union{Nothing,KeySet})
    pushrow!(tiers.buffer, row)
    k = keyvalues(row, keynames)
    admitwinrunning!(tiers.running, tiers.runprotos, k, row, ks, keynames)
    admitwintree!(tiers.trees, tiers.treeprotos, k, row,
        tiers.running === nothing ? ks : nothing, keynames)
    tiers.running === nothing && tiers.trees === nothing &&
        checkdeclared(ks, row, keynames)
    return nothing
end

@inline admitwinrunning!(::Nothing, stateprotos, k, row, ks, keynames::Val) =
    nothing
@inline admitwinrunning!(rt::RunningTable, stateprotos::Tuple, k, row, ks,
    keynames::Val) = admitgroup!(rt, stateprotos, k, row, ks, keynames)

@inline admitwintree!(::Nothing, stateprotos, k, row, ks, keynames::Val) =
    nothing
@inline function admitwintree!(trees::Dict{K,SegTree{S,R,T}}, stateprotos::S,
    k, row, ks, keynames::Val) where {K,S,R,T}
    tr = get!(trees, k) do
        checkdeclared(ks, row, keynames)
        newsegtree(stateprotos, R, T)
    end
    treeappend!(tr, stateprotos, row)
    return nothing
end

# Close tick τ: evict the rows that have left its window, downdating them out of
# the running tier; move each tree's head to the window start, dropping emptied
# trees and syncing the rest; fold the live rows into the refold table; emit;
# then retire the refold states to the table's pool — `summaryvalues` has
# copied their values out by then, which is the `closecycle!` protocol.
@noinline function closewindow!(rows::Vector, τ, tiers::WindowTiers, head::Int,
    prevkeys::Vector, lookback, keynames::Val, outs::Val, emptyrow, grid::Val,
    ks::Union{Nothing,KeySet})
    head = evictwindow!(tiers.buffer, head, τ, lookback, tiers.running, keynames)
    head = compacthead!(tiers.buffer, head)
    closetrees!(tiers.trees, τ, lookback)
    foldrefold!(tiers.refold, tiers.buffer, head, tiers.refoldprotos, keynames)
    emitgroups!(rows, τ, primarygroups(tiers), tiers.scratch, prevkeys, tiers,
        outs, emptyrow, grid, ks)
    retirerefold!(tiers.refold)
    return head
end

@inline evictwindow!(::Nothing, head::Int, τ, lookback, running, keynames::Val) =
    head
@inline function evictwindow!(buffer::Vector, head::Int, τ, lookback, running,
    keynames::Val)
    while head <= length(buffer) && τ - @inbounds(buffer[head]).time > lookback
        evictwinrunning!(running, @inbounds(buffer[head]), keynames)
        head += 1
    end
    return head
end

@inline evictwinrunning!(::Nothing, row, keynames::Val) = nothing
@inline evictwinrunning!(rt::RunningTable, row, keynames::Val) =
    evictgroup!(rt, keyvalues(row, keynames), row)

# Every admitted row is before τ, so a tree's window is head:length(rows). A
# tree whose window has emptied is dropped: ticks only advance, so its rows are
# expired for good. The survivors sync the leaves appended since the last tick,
# once, before their one query.
@inline closetrees!(::Nothing, τ, lookback) = nothing
@inline function closetrees!(trees::AbstractDict, τ, lookback)
    filter!(trees) do (_, tr)
        tr.head = windowstart(tr.times, tr.head, τ, lookback)
        live = tr.head <= length(tr.rows)
        live && treesync!(tr)
        return live
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
@inline function retirerefold!(gt::GroupTable)
    for states in values(gt.table)
        push!(gt.pool, states)
    end
    empty!(gt.table)
    return nothing
end

# One tier's states for key k, whose entry in the primary tier is g: `()` for
# an absent tier, g itself for the primary (no second lookup), a lookup by key
# for the others, and for a tree its borrowed query (see `treequery`), read
# once per tick through summaryvalues. The running tier, when present, is
# always the primary.
@inline tierstates(::Nothing, k, g) = ()
@inline tierstates(::RunningTable{K,S}, k, g::RunningGroup{S}) where {K,S<:Tuple} =
    g.states
@inline tierstates(::Dict{K,V}, k, tr::V) where {K,V<:SegTree} =
    treequery(tr, tr.head, length(tr.rows))
@inline tierstates(trees::Dict{K,V}, k, g) where {K,V<:SegTree} =
    tierstates(trees, k, trees[k])
@inline tierstates(::GroupTable{K,S}, k, states::S) where {K,S<:Tuple} = states
@inline tierstates(gt::GroupTable, k, g) = gt.table[k]

@inline tiervalues(tiers::WindowTiers, k, g, outs::Val) = summaryvalues(
    mergestates(tiers.perm,
        (tierstates(tiers.running, k, g), tierstates(tiers.trees, k, g),
            tierstates(tiers.refold, k, g), tiers.derived)), outs)

# Emit one tick from the primary tier's groups. Undeclared keys: the present
# ones sorted into the reused scratch, then `emitwindow!`'s grid or vanish-row
# merge. A declared keyset: `emitdense!`, which needs neither the sort nor
# `prevkeys`.
@inline function emitgroups!(rows::Vector, τ, groups::AbstractDict,
    scratch::Vector{<:Pair}, prevkeys::Vector, tiers::WindowTiers, outs::Val,
    emptyrow, grid::Val, ::Nothing)
    empty!(scratch)
    append!(scratch, groups)
    sort!(scratch; by = groupkey)
    return emitwindow!(rows, τ, scratch, prevkeys, tiers, outs, emptyrow, grid)
end
@inline emitgroups!(rows::Vector, τ, groups::AbstractDict, scratch,
    prevkeys::Vector, tiers::WindowTiers, outs::Val, emptyrow, grid::Val,
    ks::KeySet) = emitdense!(rows, τ, groups, ks, tiers, outs, emptyrow)

# One row per declared key, in declared order: the key's window summary when it
# has a group, the empty values otherwise. The declared key type may differ from
# the groups' (it is `isequal`), which the Dict lookups allow without converting.
function emitdense!(rows::Vector{RT}, τ, groups::AbstractDict, ks::KeySet,
    tiers::WindowTiers, outs::Val, emptyrow) where {RT}
    # `haskey` then indexing, not `get(groups, k, nothing)`: a re-fold primary's
    # entry is a tuple of states, which a Union with Nothing would box per key.
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

# Emit one tick. Keyless (grid): exactly one row, the summary or the empty
# values. Keyed: the present keys in key order, merged with an empty row for
# every key present at the previous tick and absent now; the present keys then
# become the previous tick's. Both lists are sorted by the tuple of key values,
# the order `groupkey` gives, so one merge walk suffices.
function emitwindow!(rows::Vector{RT}, τ, present::Vector{<:Pair},
    prevkeys::Vector{K}, tiers::WindowTiers, outs::Val, emptyrow,
    ::Val{G}) where {RT,K,G}
    if G
        push!(rows,
            convert(RT,
                isempty(present) ? gridrow(τ, (;), emptyrow) :
                gridrow(
                    τ,
                    (;),
                    tiervalues(tiers, first(first(present)),
                        last(first(present)), outs),
                )))
        return rows
    end
    j = 1
    np = length(prevkeys)
    for (k, g) in present
        while j <= np && isless(values(@inbounds(prevkeys[j])), values(k))
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
