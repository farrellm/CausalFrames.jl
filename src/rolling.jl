# The rolling-window summarization transform. The augmented stream drives a
# chunkmap; the summarized stream is pulled on demand from inside the step.
# Each summarizer's window algorithm follows its own structure (tiers.jl):
# groups slide per-key running states in O(1) amortized per row (update!
# entering rows, downdate! exiting ones, oldest first), other monoids fold
# each window from a per-key segment tree of partial combine!s in O(log
# window), and anything less re-folds fresh states over a buffer of the window
# rows, O(window) per row. One kernel drives all three tiers, emitting each
# window from their states merged back into dependency order. Per the
# summarize.jl conventions, the type-unstable setup happens once per chunk and
# the kernel takes concretely typed arguments behind a function barrier.

"""
    addrollingcolumns(windows, summarizers; key = nothing,
                      from = nothing) -> (CausalPipeline -> CausalPipeline)
    addrollingcolumns(p::CausalPipeline, windows, summarizers;
                      ...) -> CausalPipeline

A transform appending, for each row at time `t` and each window, the summaries
of the rows with time in `[t - lookback, t]`. Columns are named
`{window}_{column}`, e.g. `m5_price_sum`. The summarized pipeline runs over a
context widened by the longest look-back, so the first row already sees a full
window.

# Arguments
- `windows`: window names and their look-backs, as a `NamedTuple`
  (`(m5 = Minute(5), h1 = Hour(1))`), a `name => lookback` pair, or a collection
  of pairs. Names must be unique; look-backs must be non-negative and
  subtractable from the time type.
- `summarizers`: a [`Summarizer`](@ref) or a collection of them. Their output
  columns may not collide with existing columns.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present in both inputs. With a key, a row's window holds only
  rows with the same key.
- `from = nothing`: a pipeline to summarize instead of the input itself. Its
  rows relate to the output rows by time and key only. By default the input is
  summarized, so it runs twice.

An empty window, including one for an unseen key, gives the summarizers' empty
values, so a column's type may widen (`Min` gives `Union{Missing, T}`).
"""
function addrollingcolumns(windows, summarizers; key = nothing,
    from::Union{Nothing,CausalPipeline} = nothing)
    windownames, lookbacks = towindows(windows)
    isempty(windownames) &&
        throw(ArgumentError("addrollingcolumns requires at least one window"))
    allunique(windownames) ||
        throw(ArgumentError("addrollingcolumns window names must be unique"))
    keycols = keycolumns(key, "addrollingcolumns")
    protos, requested =
        prototypes(tosummarizers(summarizers), Symbol[], "addrollingcolumns")
    prefixednames = Symbol[]
    for w in windownames, n in requested
        pn = Symbol(w, '_', n)
        pn in prefixednames && throw(
            ArgumentError(
                "addrollingcolumns output column $(repr(pn)) appears more than once"),
        )
        push!(prefixednames, pn)
    end
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            source = from === nothing ? p : from
            cfg = RollingConfig(windownames, lookbacks, keycols,
                Val(Tuple(keycols)), protos, Val(requested),
                prefixednames)
            rs = RollingState(source.run(rollingcontext(ctx, lookbacks)))
            return chunkmap(c -> rollchunk!(rs, cfg, c), p.run(ctx))
        end
    end
end
addrollingcolumns(p::CausalPipeline, windows, summarizers; kwargs...) =
    addrollingcolumns(windows, summarizers; kwargs...)(p)

# Normalized window spec: names as a Symbol tuple, look-backs as a tuple so
# the possibly heterogeneous look-back types (say Minute and Hour) stay
# concrete into the kernel.
towindows(w::NamedTuple) = (keys(w), values(w))
towindows(w::Pair) = ((Symbol(first(w)),), (last(w),))
function towindows(w)
    ps = collect(w)
    all(p -> p isa Pair, ps) || throw(
        ArgumentError(
            "addrollingcolumns windows must be a NamedTuple or name => lookback pairs"),
    )
    return (Tuple(Symbol(first(p)) for p in ps), Tuple(last(p) for p in ps))
end

# Look-backs are never compared to each other (they may be of incomparable
# types); the widened starts all live in the time type, so the earliest one
# is found there.
function rollingcontext(ctx::Context, lookbacks::Tuple)
    starts = map(lb -> widenstart(ctx, lb, "addrollingcolumns lookback").start,
        lookbacks)
    return Context(minimum(starts), ctx.stop)
end

struct RollingConfig{KN,LB<:Tuple,P<:Tuple,O}
    windownames::Tuple{Vararg{Symbol}}
    lookbacks::LB
    keycols::Vector{Symbol}
    keynames::Val{KN}
    protos::P
    outs::Val{O}
    prefixednames::Vector{Symbol}
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the rollsegment! function barrier.
mutable struct RollingState
    const summarized::PullCursor  # the summarized chunks; `done` once exhausted
    snt::Any           # current summarized column table (nothing until pulled)
    spos::Int          # index of the next unadmitted summarized row in snt
    stypes::Union{Nothing,NamedTuple}  # promotion of summarized schemas seen
    tiers::Any         # RollTiers for the realized state types
    valtype::Any       # output value NamedTuple type (summary ∪ empty values)
    emptyrow::Any      # emptyvalues converted to valtype
    vals::Any          # per-window value vectors for the chunk in progress
    passthrough::Bool  # the summarized stream produced no chunks at all
    checked::Bool      # augmented-side name/key validation done
    RollingState(schunks) = new(PullCursor(schunks), nothing, 1,
        nothing, nothing, nothing, nothing, nothing, false, false)
end

# The window structures for one tiering (tiers.jl), R being the stored row
# type. The buffer holds every admitted row still inside some window, in time
# order, with a per-window eviction head; the running tier downdates rows as
# its window's head passes them, and the refold tier folds each window from
# its head. The trees own their rows, so a call with only tree and derived
# states keeps no buffer (`nothing`); one mixing the tree with another tier
# stores its rows twice.
struct RollTiers{R,B,G,TR,SR<:Tuple,ST<:Tuple,SF<:Tuple,SD<:Tuple,P}
    buffer::B              # Vector{R}, or nothing
    winheads::Vector{Int}  # per window, the first buffered row inside it
    running::G             # NTuple{W,RunningTable{K,SR}}, or nothing
    trees::TR              # Dict{K,SegTree{ST,R,T}}, or nothing
    runprotos::SR
    treeprotos::ST
    refoldprotos::SF
    derived::SD
    perm::Val{P}
end

# Build the tiers for the realized types, from scratch or — after a widening —
# from the old tiers' live rows, always rebuilding rather than widening in
# place: rare, O(live), and correct for every transition, including an
# accumulator the widening demotes from the running tier to the tree. The trees
# replay from the old trees when there were any, from the buffer when they are
# new; a buffer the old tiers did not keep is gathered from the old trees (see
# `treebuffer`). Type-unstable, once per run and per widening.
function rolltiers(tg::Tiering, types::NamedTuple, keynames::Val, nwindows::Int,
    old)
    K = storekeytype(types, keynames)
    R = storerowtype(types)
    oldbuffered = old !== nothing && old.buffer !== nothing
    buffer = rebuildbuffer(tg, R, old)
    winheads =
        buffer === nothing ? Int[] :
        oldbuffered ? copy(old.winheads) : ones(Int, nwindows)
    running =
        isempty(tg.running) ? nothing :
        ntuple(
            w -> replayrunning!(RunningTable{K,typeof(tg.running)}(),
                buffer, winheads[w], tg.running, keynames), nwindows)
    trees = rebuildtrees(tg, K, R, types.time, old,
        oldbuffered ? minimum(old.winheads) : 1, keynames)
    return RollTiers{R}(buffer, winheads, running, trees, tg.running, tg.tree,
        tg.refold, tg.derived, tg.perm)
end

RollTiers{R}(buffer::B, winheads::Vector{Int}, running::G, trees::TR,
    runprotos::SR, treeprotos::ST, refoldprotos::SF, derived::SD,
    perm::Val{P}) where {R,B,G,TR,SR,ST,SF,SD,P} =
    RollTiers{R,B,G,TR,SR,ST,SF,SD,P}(buffer, winheads, running, trees,
        runprotos, treeprotos, refoldprotos, derived, perm)

# Pull the next summarized chunk (type-unstable, once per summarized chunk):
# build the tiers on the first one and rebuild them when the promoted schema
# moves, re-typing any half-filled value vectors along the way.
function pullsummarized!(rs::RollingState, cfg::RollingConfig)
    chunk = pull!(rs.summarized)
    if chunk === nothing
        rs.snt === nothing && (rs.passthrough = true)
        return nothing
    end
    rs.stypes === nothing && checkkeycolumns(cfg.keycols, chunk,
        "addrollingcolumns", "the summarized input")
    types = promotetypes(rs.stypes, chunktypes(chunk))
    moved = rs.stypes === nothing || types != rs.stypes
    rs.stypes = types
    rs.snt = Tables.columntable(chunk)
    rs.spos = 1
    if moved
        tg = tiering(cfg.protos, types)
        rs.tiers = rolltiers(tg, types, cfg.keynames, length(cfg.lookbacks),
            rs.tiers)
        setvaltype!(rs, cfg, tg)
    end
    return nothing
end

# The output value type (summary values promoted field-wise with the empty
# values an empty window emits), cached with the converted empty row; any
# half-filled value vectors are re-typed.
function setvaltype!(rs::RollingState, cfg::RollingConfig, tg::Tiering)
    V = tieredvaluetype(tg, cfg.protos, cfg.outs)
    rs.valtype = V
    rs.emptyrow = convert(V, emptyvalues(cfg.protos, cfg.outs))
    rs.vals === nothing ||
        (rs.vals = map(v -> convert(Vector{V}, v), rs.vals))
    return nothing
end

function rollchunk!(rs::RollingState, cfg::RollingConfig, c::DataFrame)
    if !rs.checked
        checkkeycolumns(cfg.keycols, c, "addrollingcolumns", "the augmented input")
        for pn in cfg.prefixednames
            String(pn) in names(c) && throw(
                ArgumentError(
                    "addrollingcolumns output column $(repr(pn)) collides with an existing column",
                ),
            )
        end
        rs.checked = true
    end
    rs.snt === nothing && !rs.summarized.done && pullsummarized!(rs, cfg)
    rs.passthrough && return assembleempty(cfg, c)
    lnt = Tables.columntable(c)
    # Pre-filled with the empty row so a mid-chunk widen never converts an
    # undef slot.
    rs.vals = map(_ -> newvals(rs.valtype, rs.emptyrow, nrow(c)),
        cfg.lookbacks)
    i = 1
    # The tiers are re-read each pass: a widening inside pullsummarized!
    # rebuilds them, possibly with a different partition, mid-chunk.
    while true
        i, rs.spos, needpull = rollsegment!(rs.vals, rs.tiers, lnt, i, rs.snt,
            rs.spos, rs.summarized.done, cfg.lookbacks, cfg.keynames, cfg.outs,
            rs.emptyrow)
        needpull || break
        pullsummarized!(rs, cfg)
    end
    return assemble(cfg, rs, c)
end

newvals(::Type{V}, emptyrow, n::Int) where {V} =
    fill!(Vector{V}(undef, n), emptyrow)

# Dead rows accumulate at the front of the buffer as the heads advance; a row
# is dead once the laggiest window's head has passed it. Dropping them only when
# they dominate keeps the cost amortized O(1) per admitted row.
@inline compactbuffer!(::Nothing, winheads::Vector{Int}) = nothing
@inline function compactbuffer!(buffer::Vector{R}, winheads::Vector{Int}) where {R}
    dead = minimum(winheads) - 1
    if dead >= 64 && 2 * dead >= length(buffer)
        deleteat!(buffer, 1:dead)
        winheads .-= dead
    end
    return nothing
end

# --- the kernel ------------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access, and an absent tier to nothing at all. Processes
# augmented rows from index i, admitting summarized rows from snt starting at
# spos. Returns (i, spos, needpull): needpull means the current summarized
# chunk is consumed but the stream may still hold rows with time <= row i's
# time — the driver must pull the next summarized chunk before row i can be
# emitted.
#
# Per augmented row at time t: admit the summarized rows not after t (equal
# times included, so every row tied at t is in before any row at t is emitted)
# into every tier; advance each window's eviction head past the rows older than
# its look-back, downdating them out of the running tier; then emit each window
# from the three tiers' states for the row's key.
function rollsegment!(vals::Tuple, tiers::RollTiers{R}, lnt::NamedTuple, i::Int,
    snt::NamedTuple, spos::Int, sdone::Bool, lookbacks::Tuple,
    keynames::Val, outs::Val, emptyrow) where {R}
    n = length(lnt.time)
    slen = length(snt.time)
    # One refold state tuple for the whole segment: every window of every row
    # folds into it and `summaryvalues` copies the result out before it is
    # zeroed again, so re-folding allocates per segment, not per row.
    scratch = map(fresh, tiers.refoldprotos)
    while i <= n
        t = @inbounds lnt.time[i]
        while spos <= slen && @inbounds(snt.time[spos]) <= t
            admitroll!(tiers, rowat(R, snt, spos), keynames)
            spos += 1
        end
        spos > slen && !sdone && return (i, spos, true)
        # Times are non-decreasing, so a row outside a window now is outside it
        # forever. A row admitted already outside a short window enters its
        # running group and is downdated right back out here.
        evictroll!(t, tiers, tiers.buffer, keynames, 1, lookbacks...)
        # Compacting here rather than once per chunk keeps the buffer at the
        # window's size instead of the chunk's, which is what its growth costs.
        compactbuffer!(tiers.buffer, tiers.winheads)
        k = keyat(lnt, i, keynames)
        tr = keytree(tiers.trees, k)
        scratch, minlo = emitroll!(vals, i, t, k, tiers, tr, scratch, keynames,
            outs, emptyrow, typemax(Int), 1, lookbacks...)
        advancetree!(tr, minlo)
        i += 1
    end
    return (i, spos, false)
end

@inline function admitroll!(tiers::RollTiers, row, keynames::Val)
    pushrow!(tiers.buffer, row)
    k = keyvalues(row, keynames)
    admitrunning!(tiers.running, tiers.runprotos, k, row)
    admittree!(tiers.trees, tiers.treeprotos, k, row)
    return nothing
end

# The per-window tables are a homogeneous tuple, so indexing it by an Int is
# type-stable.
@inline admitrunning!(::Nothing, stateprotos, k, row) = nothing
@inline function admitrunning!(tables::Tuple, stateprotos::Tuple, k, row)
    for w in 1:length(tables)
        admitgroup!(tables[w], stateprotos, k, row)
    end
    return nothing
end

@inline admittree!(::Nothing, stateprotos, k, row) = nothing
@inline admittree!(trees::Dict{K,SegTree{S,R,T}}, stateprotos::S, k,
    row) where {K,S,R,T} =
    treepush!(get!(() -> newsegtree(stateprotos, R, T), trees, k), stateprotos,
        row)

# One window per recursion step, peeling the (possibly heterogeneous)
# look-backs with the window index in tow.
@inline evictroll!(t, tiers::RollTiers, ::Nothing, keynames::Val, w::Int,
    lookbacks...) = nothing
@inline evictroll!(t, tiers::RollTiers, buffer::Vector, keynames::Val,
    w::Int) = nothing
@inline function evictroll!(t, tiers::RollTiers, buffer::Vector, keynames::Val,
    w::Int, lb, rest...)
    head = @inbounds tiers.winheads[w]
    while head <= length(buffer) && t - @inbounds(buffer[head]).time > lb
        evictrunning!(tiers.running, w, @inbounds(buffer[head]), keynames)
        head += 1
    end
    @inbounds tiers.winheads[w] = head
    return evictroll!(t, tiers, buffer, keynames, w + 1, rest...)
end

@inline evictrunning!(::Nothing, w::Int, row, keynames::Val) = nothing
@inline evictrunning!(tables::Tuple, w::Int, row, keynames::Val) =
    evictgroup!(tables[w], keyvalues(row, keynames), row)

# A tier's states for one window: `()` when the call has no such tier, `nothing`
# when the window holds no rows of the key, the state tuple otherwise.
@inline keytree(::Nothing, k) = ()
@inline keytree(trees::AbstractDict, k) = get(trees, k, nothing)

@inline runningstates(::Nothing, w::Int, k) = ()
@inline function runningstates(tables::Tuple, w::Int, k)
    g = get(tables[w].groups, k, nothing)
    return g === nothing ? nothing : g.states
end

# The window's first live row, searched with the rolling membership predicate
# verbatim (segtree.jl's `windowstart`), and the borrowed query result.
@inline treestates(::Tuple{}, t, lb) = ((), typemax(Int))
@inline treestates(::Nothing, t, lb) = (nothing, typemax(Int))
@inline function treestates(tr::SegTree, t, lb)
    lo = windowstart(tr.times, tr.head, t, lb)
    hi = length(tr.rows)
    return (lo > hi ? nothing : treequery(tr, lo, hi)), lo
end

# The row minimum of the window starts advances the tree's head: a row older
# than every window is expired for good, and the tree drops it at its next
# rebuild.
@inline advancetree!(tr, minlo::Int) = nothing
@inline advancetree!(tr::SegTree, minlo::Int) = (tr.head = max(tr.head, minlo); nothing)

# Fold the window's rows of key k, from its eviction head, into the zeroed
# scratch — the rows past the head are exactly the window's. Returns the states
# (or nothing, for no rows) and the scratch to thread on.
@inline refoldstates(scratch::Tuple{}, buffer, winheads::Vector{Int}, w::Int, k,
    keynames::Val) = ((), scratch)
@inline function refoldstates(scratch::Tuple{Any,Vararg}, buffer::Vector,
    winheads::Vector{Int}, w::Int, k, keynames::Val)
    states = freshall!(scratch)
    seen = false
    for j in (@inbounds winheads[w]):length(buffer)
        s = @inbounds buffer[j]
        isequal(keyvalues(s, keynames), k) || continue
        updateall!(states, s)
        seen = true
    end
    return (seen ? states : nothing), states
end

# One window per recursion step. Every tier holds the same rows per key, so a
# `nothing` from any of them is the empty window; otherwise the tiers' states
# are merged back into topological order and summarized.
@inline emitroll!(vals::Tuple, i::Int, t, k, tiers::RollTiers, tr, scratch,
    keynames::Val, outs::Val, emptyrow, minlo::Int, w::Int) = (scratch, minlo)
@inline function emitroll!(vals::Tuple, i::Int, t, k, tiers::RollTiers, tr,
    scratch, keynames::Val, outs::Val, emptyrow, minlo::Int, w::Int, lb,
    rest...)
    rs = runningstates(tiers.running, w, k)
    ts, lo = treestates(tr, t, lb)
    fs, scratch = refoldstates(scratch, tiers.buffer, tiers.winheads, w, k,
        keynames)
    v = vals[w]
    if rs === nothing || ts === nothing || fs === nothing
        @inbounds v[i] = emptyrow
    else
        @inbounds v[i] = summaryvalues(
            mergestates(tiers.perm, (rs, ts, fs, tiers.derived)), outs)
    end
    return emitroll!(vals, i, t, k, tiers, tr, scratch, keynames, outs,
        emptyrow, min(minlo, lo), w + 1, rest...)
end

# --- output assembly -------------------------------------------------------

function assemble(cfg::RollingConfig, rs::RollingState, c::DataFrame)
    for (wname, v) in zip(cfg.windownames, rs.vals)
        wdf = DataFrame(v)
        rename!(n -> string(wname, '_', n), wdf)
        # The chunk is owned, so columns can be adopted rather than copied.
        c = hcat(c, wdf; copycols = false)
    end
    return c
end

# A summarized stream with no chunks at all: every window is empty, so every
# row gets the empty values, typed from the summarizer configs alone.
function assembleempty(cfg::RollingConfig, c::DataFrame)
    e = emptyvalues(cfg.protos, cfg.outs)
    for wname in cfg.windownames, (name, val) in pairs(e)
        c[!, Symbol(wname, '_', name)] = fill(val, nrow(c))
    end
    return c
end
