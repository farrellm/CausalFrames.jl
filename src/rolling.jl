# The rolling-window summarization transform. The augmented stream drives a
# chunkmap and pulls the summarized stream on demand. Each accumulator's window
# algorithm follows its structure (tiers.jl): groups slide per-key running
# states in O(1) amortized per row, other monoids query a per-key segment tree
# in O(log window), and the rest re-fold the window's buffered rows in
# O(window). One kernel drives all three tiers, emitting each window from their
# states merged back into dependency order. Type-unstable setup runs once per
# chunk; the kernel takes concretely typed arguments. A `Bars` window counts
# rows instead: each key keeps a ring of its newest rows, and a row leaves the
# window when it is pushed `n` rows deep, on admission rather than by time.

"""
    Bars(n::Integer) -> Bars

A bar-count look-back for [`addrollingcolumns`](@ref): the window at a row
holds the last `n` summarized rows with time at or before the row's (under
the row's key), where a time look-back holds the rows within a time span.
Until `n` rows have arrived, every output of the window is `missing`, even a
summarizer's whose empty value is not ([`Count`](@ref), [`Sum`](@ref)), so
each of its columns is `Union{Missing, T}`.

Rows sharing a timestamp are admitted together, so a row followed by rows at
the same time has them in its window: this is the last `n` rows per row only
when times are unique per key. A bar count has no time span, so it doesn't
widen the summarized input to before `start`, and the first `n - 1` rows of each
key are `missing`. A time look-back in the same call does widen it, and the
`Bars` window then counts those earlier rows too.

# Arguments
- `n`: the number of rows in the window; less than 1 is an `ArgumentError`.

```jldoctest
df = DataFrame(time = [1, 2, 4, 7], x = [1.0, 2.0, 3.0, 4.0])
p = readtable(df) |> addrollingcolumns((b2 = Bars(2), t2 = 2), Sum(:x))
DataFrame(load(Context(0, 10), p))

# output

4×4 DataFrame
 Row │ time   x        b2_x_sum   t2_x_sum
     │ Int64  Float64  Float64?   Float64
─────┼─────────────────────────────────────
   1 │     1      1.0  missing         1.0
   2 │     2      2.0        3.0       3.0
   3 │     4      3.0        5.0       5.0
   4 │     7      4.0        7.0       4.0
```
"""
struct Bars
    n::Int
    function Bars(n::Integer)
        n >= 1 || throw(ArgumentError("Bars count must be positive, got $n"))
        return new(Int(n))
    end
end

"""
    addrollingcolumns(windows, summarizers; key = nothing, from = nothing,
                      sharedrun = true) -> (CausalPipeline -> CausalPipeline)
    addrollingcolumns(p::CausalPipeline, windows, summarizers;
                      ...) -> CausalPipeline

A transform appending, for each row at time `t` and each window, the summaries
of the rows with time in `[t - lookback, t]`, or of the last `n` rows under a
[`Bars`](@ref)`(n)` look-back. Columns are named `{window}_{column}`, e.g.
`m5_price_sum`. The summarized pipeline runs over a context widened by the
longest time look-back, so the first row already sees a full window.

# Arguments
- `windows`: window names and their look-backs, as a `NamedTuple`
  (`(m5 = Minute(5), h1 = Hour(1), b20 = Bars(20))`), a `name => lookback`
  pair, or a collection of pairs. Names must be unique; a time look-back must
  be non-negative and subtractable from the time type.
- `summarizers`: a [`Summarizer`](@ref) or a collection of them. Their output
  columns may not collide with existing columns.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present in both inputs. With a key, a row's window holds only
  rows with the same key.
- `from = nothing`: a pipeline to summarize instead of the input itself. Its
  rows relate to the output rows by time and key only. By default the input is
  summarized.
- `sharedrun = true`: without `from`, run the input once, over the widened
  context, and drop the rows before `start` from the output, so the output
  rows are the rows summarized. `false` takes the output rows from a run over
  the window itself, as a pipeline without this transform would give them; a
  time look-back then runs the input twice, once per context. The two agree
  without a time look-back, and on an input whose rows at or after `start`
  depend neither on earlier rows nor on where its run starts. They differ in
  the rows themselves under an input anchored to its run's start, such as
  [`clock`](@ref) ticks, an [`intervalize`](@ref) grid or [`head`](@ref): pass
  `false` there. Ignored with `from`.

A running count depends on where its run starts. The `w2` look-back widens
the context to start at 1, so by default the output shows the counts from
there, the ones summed:

```jldoctest sharedrun
p = clock(1) |> addsummarycolumns(Count())
t = addrollingcolumns((w2 = 2,), Sum(:count))
DataFrame(load(Context(3, 6), p |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      3             6
   2 │     4      4             9
   3 │     5      5            12
```

With `sharedrun = false` the output rows come from a run starting at 3, so
their counts restart at 1, while the summaries still sum the widened run's:

```jldoctest sharedrun
t = addrollingcolumns((w2 = 2,), Sum(:count); sharedrun = false)
DataFrame(load(Context(3, 6), p |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      1             6
   2 │     4      2             9
   3 │     5      3            12
```

An empty time window, including one for an unseen key, gives the summarizers'
empty values, so a column's type may widen (`Min` gives `Union{Missing, T}`).
A `Bars` window holding fewer than `n` rows gives `missing` instead.
"""
function addrollingcolumns(windows, summarizers; key = nothing,
    from::Union{Nothing,CausalPipeline} = nothing, sharedrun::Bool = true)
    windownames, lookbacks = towindows(windows)
    isempty(windownames) &&
        throw(ArgumentError("addrollingcolumns requires at least one window"))
    allunique(windownames) ||
        throw(ArgumentError("addrollingcolumns window names must be unique"))
    keycols = keycolumns(key, "addrollingcolumns")
    protos, requested, terms =
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
            cfg = RollingConfig(windownames, lookbacks, keycols,
                Val(Tuple(keycols)), protos, Val(requested),
                prefixednames, terms)
            summarized, augmented = rollinginputs(p, from, sharedrun, ctx,
                rollingcontext(ctx, lookbacks))
            rs = RollingState(summarized)
            return chunkmap(c -> rollchunk!(rs, cfg, c), augmented)
        end
    end
end

# The summarized and augmented chunk streams. Summarizing its own input, the
# transform runs it once and tees it, since a run per side would make a chain
# of d stages run its source 2^d times (issue #98). Over a widened context the
# augmented side drops the lead-in, as `warmup` does, unless `sharedrun` is
# off, when it keeps its own run over `ctx`.
function rollinginputs(p::CausalPipeline, from, sharedrun::Bool, ctx::Context,
    sctx::Context)
    from === nothing || return from.run(sctx), p.run(ctx)
    widened = sctx.start != ctx.start
    widened && !sharedrun && return p.run(sctx), p.run(ctx)
    summarized, augmented = teesides(p.run(sctx))
    widened || return summarized, augmented
    drop = LeadInDrop(ctx.start)
    return summarized, chunkmap(c -> dropleadin!(drop, c), augmented)
end
addrollingcolumns(p::CausalPipeline, windows, summarizers; kwargs...) =
    addrollingcolumns(windows, summarizers; kwargs...)(p)

# Normalized window spec: names as a Symbol tuple and look-backs as a tuple, so
# heterogeneous look-back types (say Minute and Hour) stay concrete.
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

# Look-backs may be of incomparable types, so the earliest widened start is
# found in the time type. A bar count has no span and widens nothing.
function rollingcontext(ctx::Context, lookbacks::Tuple)
    starts = map(lb -> lookbackstart(ctx, lb), lookbacks)
    return Context(minimum(starts), ctx.stop)
end

lookbackstart(ctx::Context, ::Bars) = ctx.start
lookbackstart(ctx::Context, lb) =
    widenstart(ctx, lb, "addrollingcolumns lookback").start

isbars(lb) = lb isa Bars

struct RollingConfig{KN,LB<:Tuple,P<:Tuple,O,TM<:NamedTuple}
    windownames::Tuple{Vararg{Symbol}}
    lookbacks::LB
    keycols::Vector{Symbol}
    keynames::Val{KN}
    protos::P
    outs::Val{O}
    prefixednames::Vector{Symbol}
    terms::TM          # the row terms, from `prototypes`
end

# Per-run state. The untyped fields are per-chunk setup; per-row work sits
# behind the rollsegment! function barrier.
mutable struct RollingState
    const summarized::PullCursor  # the summarized chunks; `done` once exhausted
    snt::Any           # current summarized column table (nothing until pulled)
    spos::Int          # index of the next unadmitted summarized row in snt
    stypes::Union{Nothing,NamedTuple}  # promotion of summarized schemas seen
    tiers::Any         # RollTiers for the realized state types
    valtypes::Any      # per window, the output value NamedTuple type
    emptyrows::Any     # per window, its empty row: emptyvalues, or all missing
    vals::Any          # per-window value vectors for the chunk in progress
    passthrough::Bool  # the summarized stream produced no chunks at all
    checked::Bool      # augmented-side name/key validation done
    RollingState(schunks) = new(PullCursor(schunks), nothing, 1,
        nothing, nothing, nothing, nothing, nothing, false, false)
end

# The window structures for one tiering (tiers.jl), R being the stored row
# type. The buffer holds the admitted rows still inside some time window, in
# time order, with a per-window eviction head: the running tier downdates rows
# as the head passes them, and the refold tier folds each window from its head.
# A `Bars` window's head sits past the buffer's end, so it never holds rows
# there. Instead each key keeps a ring of its newest `barcap` rows (one more
# than the longest bar count, so the row leaving is still there to downdate),
# from which the running tier downdates and the refold tier folds. Trees own
# their rows, so a call with only tree and derived states keeps no buffer and
# no rings (`nothing`), nor does a call without time windows keep a buffer.
struct RollTiers{R,B,RG,G,TR,SR<:Tuple,ST<:Tuple,SF<:Tuple,SD<:Tuple,P}
    buffer::B              # Vector{R}, or nothing
    winheads::Vector{Int}  # per window, the first buffered row inside it
    rings::RG              # Dict{K,RowRing{R}}, or nothing
    barcap::Int            # the rings' capacity
    running::G             # NTuple{W,RunningTable{K,SR}}, or nothing
    trees::TR              # Dict{K,SegTree{ST,R,T}}, or nothing
    runprotos::SR
    treeprotos::ST
    refoldprotos::SF
    derived::SD
    perm::Val{P}
end

# Build the tiers for the realized types, from scratch or, after a widening,
# from the previous tiers' live rows (`rebuildbuffer`, `rebuildtrees`). A
# widening always rebuilds rather than widening in place: rare, O(live), and
# correct for every transition, including one that demotes an accumulator from
# the running tier to the tree. Type-unstable, once per run and per widening.
function rolltiers(tg::Tiering, types::NamedTuple, keynames::Val,
    lookbacks::Tuple, old)
    K = storekeytype(types, keynames)
    R = storerowtype(types)
    nwindows = length(lookbacks)
    anybars = any(isbars, lookbacks)
    barcap = anybars ? maximum(lb -> isbars(lb) ? lb.n : 0, lookbacks) + 1 : 0
    oldbuffered = old !== nothing && old.buffer !== nothing
    oldhead = oldbuffered ? minimum(old.winheads) : 1
    # A time window's rows are in the buffer; a Bars window's in the rings.
    buffer = all(isbars, lookbacks) ? nothing : rebuildbuffer(tg, R, old)
    winheads =
        buffer === nothing ? Int[] :
        oldbuffered ? copy(old.winheads) : ones(Int, nwindows)
    # Each key's live rows, when a Bars window needs them from the old tiers.
    suffixes =
        anybars && old !== nothing ? barsuffixes(K, R, old, oldhead, keynames) :
        nothing
    rings =
        anybars && !(isempty(tg.running) && isempty(tg.refold)) ?
        rebuildrings(K, R, barcap, suffixes) : nothing
    running =
        isempty(tg.running) ? nothing :
        ntuple(
            w -> replaywindow!(RunningTable{K,typeof(tg.running)}(), lookbacks[w],
                buffer, buffer === nothing ? 1 : winheads[w], rings, tg.running,
                keynames), nwindows)
    trees =
        suffixes === nothing ?
        rebuildtrees(tg, K, R, types.time, old, oldhead, keynames) :
        suffixtrees(tg, K, R, types.time, suffixes, keynames)
    return RollTiers{R}(buffer, winheads, rings, barcap, running, trees,
        tg.running, tg.tree, tg.refold, tg.derived, tg.perm)
end

RollTiers{R}(buffer::B, winheads::Vector{Int}, rings::RG, barcap::Int,
    running::G, trees::TR, runprotos::SR, treeprotos::ST, refoldprotos::SF,
    derived::SD, perm::Val{P}) where {R,B,RG,G,TR,SR,ST,SF,SD,P} =
    RollTiers{R,B,RG,G,TR,SR,ST,SF,SD,P}(buffer, winheads, rings, barcap,
        running, trees, runprotos, treeprotos, refoldprotos, derived, perm)

# A window's running table, replayed from the rows the window holds: a time
# window's from the buffer (from its head), a Bars window's from each key's
# ring (its newest n rows).
replaywindow!(rt::RunningTable, lb, buffer, head::Int, rings, stateprotos,
    keynames::Val) = replayrunning!(rt, buffer, head, stateprotos, keynames)
function replaywindow!(rt::RunningTable, lb::Bars, buffer, head::Int,
    rings::AbstractDict, stateprotos, keynames::Val)
    for (_, ring) in rings, j in min(ringlength(ring), lb.n):-1:1
        row = ringback(ring, j)
        admitgroup!(rt, stateprotos, keyvalues(row, keynames), row)
    end
    return rt
end

# Every key's live rows, oldest first, for a rebuild with Bars windows. Each of
# the old tiers holds a suffix of every key's stream (its ring, its tree's live
# rows, its rows in the buffer from `head`), so the longest one holds them all.
function barsuffixes(::Type{K}, ::Type{R}, old, head::Int,
    keynames::Val) where {K,R}
    suffixes = Dict{K,Vector{R}}()
    function offer!(k, rows)
        cur = get(suffixes, k, nothing)
        (cur === nothing || length(rows) > length(cur)) && (suffixes[k] = rows)
        return nothing
    end
    if old.rings !== nothing
        for (k, ring) in old.rings
            offer!(k, R[ringback(ring, j) for j in ringlength(ring):-1:1])
        end
    end
    if old.trees !== nothing
        for (k, tr) in old.trees
            offer!(k, convert(Vector{R}, tr.rows[tr.head:end]))
        end
    end
    if old.buffer !== nothing
        bykey = Dict{K,Vector{R}}()
        for j in head:length(old.buffer)
            row = convert(R, old.buffer[j])
            push!(get!(() -> R[], bykey, keyvalues(row, keynames)), row)
        end
        foreach(((k, rows),) -> offer!(k, rows), bykey)
    end
    return suffixes
end

# Each key's ring, holding the newest `cap` rows of its suffix, if any.
function rebuildrings(::Type{K}, ::Type{R}, cap::Int, suffixes) where {K,R}
    rings = Dict{K,RowRing{R}}()
    suffixes === nothing && return rings
    for (k, rows) in suffixes
        ring = RowRing{R}(cap)
        foreach(row -> ringpush!(ring, row), rows[max(1, end-cap+1):end])
        rings[k] = ring
    end
    return rings
end

# The tree tier, replayed from every key's live rows.
function suffixtrees(tg::Tiering, ::Type{K}, ::Type{R}, ::Type{T}, suffixes,
    keynames::Val) where {K,R,T}
    isempty(tg.tree) && return nothing
    trees = Dict{K,SegTree{typeof(tg.tree),R,T}}()
    for (_, rows) in suffixes
        replaytrees!(trees, rows, 1, tg.tree, keynames)
    end
    return trees
end

# Pull the next summarized chunk (type-unstable, once per chunk), building the
# tiers on the first and rebuilding them when the promoted schema changes.
function pullsummarized!(rs::RollingState, cfg::RollingConfig)
    chunk = pull!(rs.summarized)
    if chunk === nothing
        rs.snt === nothing && (rs.passthrough = true)
        return nothing
    end
    rs.stypes === nothing && checkkeycolumns(cfg.keycols, chunk,
        "addrollingcolumns", "the summarized input")
    nt = terminput(chunk, cfg.terms, "addrollingcolumns")
    types = promotetypes(rs.stypes, chunktypes(nt))
    moved = rs.stypes === nothing || types != rs.stypes
    rs.stypes = types
    rs.snt = nt
    rs.spos = 1
    if moved
        tg = tiering(cfg.protos, types)
        rs.tiers = rolltiers(tg, types, cfg.keynames, cfg.lookbacks, rs.tiers)
        setvaltype!(rs, cfg, tg)
    end
    return nothing
end

# Cache each window's output value type and empty row, re-typing any
# half-filled value vectors. A time window's empty row is the empty values; a
# Bars window's is all missing, which its type admits instead.
function setvaltype!(rs::RollingState, cfg::RollingConfig, tg::Tiering)
    V = tieredvaluetype(tg, cfg.protos, cfg.outs)
    VB = missingfields(valuetype(typeof(mergedstates(tg)), cfg.outs), cfg.outs)
    empty = convert(V, emptyvalues(cfg.protos, cfg.outs))
    partial = missingrow(VB, cfg.outs)
    rs.valtypes = map(lb -> isbars(lb) ? VB : V, cfg.lookbacks)
    rs.emptyrows = map(lb -> isbars(lb) ? partial : empty, cfg.lookbacks)
    rs.vals === nothing ||
        (rs.vals = map((v, T) -> convert(Vector{T}, v), rs.vals, rs.valtypes))
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
    # Pre-filled so a mid-chunk widening never converts an undefined slot.
    rs.vals = map((T, e) -> newvals(T, e, nrow(c)), rs.valtypes, rs.emptyrows)
    i = 1
    # The tiers and windows are re-read each pass, since pullsummarized! may
    # rebuild the tiers and re-type the value vectors.
    while true
        wins = map(tuple, cfg.lookbacks, rs.emptyrows, rs.vals)
        i, rs.spos, needpull = rollsegment!(wins, rs.tiers, lnt, i, rs.snt,
            rs.spos, rs.summarized.done, cfg.lookbacks, cfg.keynames, cfg.outs)
        needpull || break
        pullsummarized!(rs, cfg)
    end
    return assemble(cfg, rs, c)
end

newvals(::Type{V}, emptyrow, n::Int) where {V} =
    fill!(Vector{V}(undef, n), emptyrow)

# A row is dead once every window's head has passed it. Dead rows are dropped
# only once they dominate the buffer: amortized O(1) per row.
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
# Called with concretely typed arguments; an absent tier compiles away.
# Processes augmented rows from index i, admitting summarized rows from snt,
# starting at spos. Returns (i, spos, needpull): needpull means the summarized
# chunk is used up but the stream may hold more rows at or before row i's time,
# so the driver must pull the next chunk before emitting row i.
#
# Per augmented row at time t: admit the summarized rows at or before t into
# every tier (a Bars window downdates the row its admission pushes out); advance
# each time window's eviction head past rows older than its look-back,
# downdating them out of the running tier; then emit each window from the
# tiers' states for the row's key. `wins` holds each window's look-back, empty
# row and value vector, peeled together since a Bars window's value type
# differs.
function rollsegment!(wins::Tuple, tiers::RollTiers{R}, lnt::NamedTuple, i::Int,
    snt::NamedTuple, spos::Int, sdone::Bool, lookbacks::Tuple,
    keynames::Val, outs::Val) where {R}
    n = length(lnt.time)
    slen = length(snt.time)
    # One refold state tuple for the segment, zeroed between windows after
    # `summaryvalues` copies the result out, so re-folding allocates per
    # segment, not per row.
    scratch = map(fresh, tiers.refoldprotos)
    while i <= n
        t = @inbounds lnt.time[i]
        while spos <= slen && @inbounds(snt.time[spos]) <= t
            admitroll!(tiers, rowat(R, snt, spos), keynames, lookbacks)
            spos += 1
        end
        spos > slen && !sdone && return (i, spos, true)
        # A row outside a window stays outside it. A row admitted already
        # outside a short window is downdated right back out here.
        evictroll!(t, tiers, tiers.buffer, keynames, 1, lookbacks...)
        # Compacting per row keeps the buffer at the window's size, not the
        # chunk's.
        compactbuffer!(tiers.buffer, tiers.winheads)
        k = keyat(lnt, i, keynames)
        tr = keytree(tiers.trees, k)
        scratch, minlo = emitroll!(i, t, k, tiers, tr, scratch, keynames,
            outs, typemax(Int), 1, wins...)
        advancetree!(tr, minlo)
        i += 1
    end
    return (i, spos, false)
end

@inline function admitroll!(tiers::RollTiers, row, keynames::Val,
    lookbacks::Tuple)
    pushrow!(tiers.buffer, row)
    k = keyvalues(row, keynames)
    admitrunning!(tiers.running, tiers.runprotos, k, row)
    admittree!(tiers.trees, tiers.treeprotos, k, row)
    admitbars!(tiers.rings, tiers.running, tiers.barcap, k, row, lookbacks)
    return nothing
end

# Push the row into its key's ring, then downdate, from each Bars window's
# running table, the row the push took it past: n + 1 rows deep, oldest first.
@inline admitbars!(::Nothing, running, cap::Int, k, row, lookbacks) = nothing
@inline function admitbars!(rings::Dict{K,RowRing{R}}, running, cap::Int, k,
    row, lookbacks::Tuple) where {K,R}
    ring = get!(() -> RowRing{R}(cap), rings, k)
    ringpush!(ring, row)
    evictbars!(running, ring, k, 1, lookbacks...)
    return nothing
end

@inline evictbars!(running, ring::RowRing, k, w::Int) = nothing
@inline evictbars!(running, ring::RowRing, k, w::Int, lb, rest...) =
    evictbars!(running, ring, k, w + 1, rest...)
@inline function evictbars!(running, ring::RowRing, k, w::Int, lb::Bars,
    rest...)
    ringlength(ring) > lb.n &&
        evictbarrow!(running, w, k, ringback(ring, lb.n + 1))
    return evictbars!(running, ring, k, w + 1, rest...)
end

@inline evictbarrow!(::Nothing, w::Int, k, row) = nothing
@inline evictbarrow!(tables::Tuple, w::Int, k, row) = evictgroup!(tables[w], k, row)

# The per-window tables are a homogeneous tuple, so Int indexing is type-stable.
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
# A Bars window keeps its head past the buffer's end, holding no rows there.
@inline function evictroll!(t, tiers::RollTiers, buffer::Vector, keynames::Val,
    w::Int, lb::Bars, rest...)
    @inbounds tiers.winheads[w] = length(buffer) + 1
    return evictroll!(t, tiers, buffer, keynames, w + 1, rest...)
end
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

# A Bars window holding fewer than n rows is partial: `nothing`, as if empty.
@inline runningstates(::Nothing, w::Int, k, lb) = ()
@inline function runningstates(tables::Tuple, w::Int, k, lb)
    g = get(tables[w].groups, k, nothing)
    return g === nothing || partialbars(lb, g.live) ? nothing : g.states
end

@inline partialbars(lb, live::Int) = false
@inline partialbars(lb::Bars, live::Int) = live < lb.n

# The borrowed query over the window, and the window's first live row
# (`windowstart`, the rolling membership predicate verbatim).
@inline treestates(::Tuple{}, t, lb) = ((), typemax(Int))
@inline treestates(::Nothing, t, lb) = (nothing, typemax(Int))
@inline function treestates(tr::SegTree, t, lb)
    lo = windowstart(tr.times, tr.head, t, lb)
    hi = length(tr.rows)
    return (lo > hi ? nothing : treequery(tr, lo, hi)), lo
end
# Every admitted row is at or before t, so a Bars window is the key's last n.
# The head never passes it: the earliest window start advances the head, and a
# rebuild drops only rows before the head, so a key with n rows has them all.
@inline function treestates(tr::SegTree, t, lb::Bars)
    hi = length(tr.rows)
    lo = hi - lb.n + 1
    return lo < 1 ? (nothing, 1) : (treequery(tr, lo, hi), lo)
end

# The earliest window start advances the tree's head: a row older than every
# window is expired, and the next rebuild drops it.
@inline advancetree!(tr, minlo::Int) = nothing
@inline advancetree!(tr::SegTree, minlo::Int) = (tr.head = max(tr.head, minlo); nothing)

# Fold key k's rows from the window's eviction head into the zeroed scratch
# (a Bars window: the newest n rows of the key's ring). Returns the states
# (nothing for no rows, or a partial Bars window) and the scratch to thread on.
@inline refoldstates(scratch::Tuple{}, tiers::RollTiers, w::Int, k,
    keynames::Val, lb) = ((), scratch)
@inline refoldstates(scratch::Tuple{Any,Vararg}, tiers::RollTiers, w::Int, k,
    keynames::Val, lb) =
    refoldbuffer(scratch, tiers.buffer, tiers.winheads, w, k, keynames)
@inline function refoldstates(scratch::Tuple{Any,Vararg}, tiers::RollTiers,
    w::Int, k, keynames::Val, lb::Bars)
    ring = get(tiers.rings, k, nothing)
    (ring === nothing || ringlength(ring) < lb.n) && return (nothing, scratch)
    states = freshall!(scratch)
    for j in (lb.n):-1:1
        updateall!(states, ringback(ring, j))
    end
    return states, states
end

@inline function refoldbuffer(scratch::Tuple, buffer::Vector,
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

# One window per recursion step, each a (look-back, empty row, value vector).
# A `nothing` from any tier is the empty (or partial) window; otherwise the
# tiers' states are merged into topological order and summarized.
@inline emitroll!(i::Int, t, k, tiers::RollTiers, tr, scratch, keynames::Val,
    outs::Val, minlo::Int, w::Int) = (scratch, minlo)
@inline function emitroll!(i::Int, t, k, tiers::RollTiers, tr, scratch,
    keynames::Val, outs::Val, minlo::Int, w::Int, win::Tuple, rest...)
    lb, emptyrow, v = win
    rs = runningstates(tiers.running, w, k, lb)
    ts, lo = treestates(tr, t, lb)
    fs, scratch = refoldstates(scratch, tiers, w, k, keynames, lb)
    if rs === nothing || ts === nothing || fs === nothing
        @inbounds v[i] = emptyrow
    else
        @inbounds v[i] = summaryvalues(
            mergestates(tiers.perm, (rs, ts, fs, tiers.derived)), outs)
    end
    return emitroll!(i, t, k, tiers, tr, scratch, keynames, outs,
        min(minlo, lo), w + 1, rest...)
end

# --- output assembly -------------------------------------------------------

function assemble(cfg::RollingConfig, rs::RollingState, c::DataFrame)
    for (wname, v) in zip(cfg.windownames, rs.vals)
        wdf = DataFrame(v)
        rename!(n -> string(wname, '_', n), wdf)
        # The chunk is owned, so its columns are adopted, not copied.
        c = hcat(c, wdf; copycols = false)
    end
    return c
end

# A summarized stream with no chunks: every window is empty, so every row gets
# the empty values, or missing under a Bars window.
function assembleempty(cfg::RollingConfig, c::DataFrame)
    e = emptyvalues(cfg.protos, cfg.outs)
    for (wname, lb) in zip(cfg.windownames, cfg.lookbacks), (name, val) in pairs(e)
        c[!, Symbol(wname, '_', name)] =
            isbars(lb) ? Vector{Union{Missing,typeof(val)}}(missing, nrow(c)) :
            fill(val, nrow(c))
    end
    return c
end
