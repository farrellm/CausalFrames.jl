# The rolling-window summarization transform. The augmented stream drives a
# chunkmap; the summarized stream is pulled on demand from inside the step.
# The window algorithm follows the summarizers' structure, classified over
# the expanded prototype tuple: all GroupSummarizers slide per-key running
# states in O(1) amortized per row (update! entering rows, downdate!
# exiting ones); all MonoidSummarizers fold each window from a per-key
# segment tree of partial combine!s in O(log window); anything less re-folds
# fresh states over a buffer of the window rows, O(window) per row. The
# running mode additionally demotes to the tree when the realized
# accumulator types defeat downdate! (isinvertible) — a widening can
# introduce that mid-stream. Per the summarize.jl conventions, the
# type-unstable setup happens once per chunk and the folding kernels take
# concretely typed arguments behind function barriers.

"""
    addrollingcolumns(windows, summarizers; key = nothing,
                      from = nothing) -> (CausalPipeline -> CausalPipeline)
    addrollingcolumns(p::CausalPipeline, windows, summarizers;
                      ...) -> CausalPipeline

A transform keeping all existing columns and appending, for each named
window, each summarizer's value columns computed over that row's trailing
window: the summarized rows with time `s` satisfying `s <= t` and
`t - s <= lookback`, inclusive on both ends, for an output row at time `t`.
`windows` maps window names to look-backs — a NamedTuple
(`(m5 = Minute(5), h1 = Hour(1))`), a single `name => lookback` pair, or a
collection of pairs — and each summary column is named by prefixing the
window name: `m5_price_sum`. Look-backs must be non-negative; as for
`asofjoin` tolerance, the time type must support subtraction (numbers and
`Dates` types do).

By default the summaries are computed over the pipeline being augmented,
which then runs twice — once for the rows, once for the summaries. `from`
names a different pipeline to summarize instead; its rows relate to the
output rows by time (and key) only, never by row identity. Either way the
summarized pipeline runs over the widened context
`[minimum over windows of start - lookback, stop)`, so windows reaching
before `start` are fully covered and the first output row already sees a
full look-back of history.

With `key` (a column name or collection of column names, present in both
inputs) each row's window holds only the summarized rows sharing its key
value. A window holding no rows — including a key never seen — yields the
summarizers' empty values, so an output column's element type widens
accordingly (e.g. `Min` gives `Union{Missing, T}`). A summarized stream
producing no chunks yields the empty values everywhere. The output columns
may not collide with existing columns.

The curried form composes with `|>`; the uncurried form applies directly,
so `addrollingcolumns(p, w, ss; ...)` is equivalent to
`p |> addrollingcolumns(w, ss; ...)`.
"""
function addrollingcolumns(windows, summarizers; key = nothing,
                           from::Union{Nothing,CausalPipeline} = nothing)
    windownames, lookbacks = towindows(windows)
    isempty(windownames) &&
        throw(ArgumentError("at least one window is required"))
    allunique(windownames) ||
        throw(ArgumentError("addrollingcolumns window names must be unique"))
    keycols = tokeycolumns(key)
    allunique(keycols) ||
        throw(ArgumentError("addrollingcolumns key columns must be unique"))
    :time in keycols && throw(ArgumentError(
        "time is the window dimension and may not be an addrollingcolumns key"))
    protos, requested = prototypes(tosummarizers(summarizers), Symbol[])
    prefixednames = Symbol[]
    for w in windownames, n in requested
        pn = Symbol(w, '_', n)
        pn in prefixednames && throw(ArgumentError(
            "addrollingcolumns output column $pn appears more than once"))
        push!(prefixednames, pn)
    end
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            source = from === nothing ? p : from
            cfg = RollingConfig(windownames, lookbacks, keycols,
                                Val(Tuple(keycols)), protos, Val(requested),
                                prefixednames, candidatemode(protos))
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
    all(p -> p isa Pair, ps) || throw(ArgumentError(
        "addrollingcolumns windows must be a NamedTuple or name => lookback pairs"))
    return (Tuple(Symbol(first(p)) for p in ps), Tuple(last(p) for p in ps))
end

# Look-backs are never compared to each other (they may be of incomparable
# types); the widened starts all live in the time type, so the earliest one
# is found there, and non-negativity falls out of `start - lb <= start`.
function rollingcontext(ctx::Context, lookbacks::Tuple)
    starts = map(lb -> ctx.start - lb, lookbacks)
    for (lb, s) in zip(lookbacks, starts)
        s <= ctx.start || throw(ArgumentError(
            "addrollingcolumns lookback must be non-negative, got $lb"))
    end
    return Context(minimum(starts), ctx.stop)
end

# Window-algorithm selection. The candidate mode is classified over the
# expanded prototype tuple at construction time; the effective mode is
# re-derived from the realized states whenever they are built or widened,
# because a Missing-widened accumulator absorbs and so defeats downdate! —
# the running mode then demotes to the tree, whose queries never combine an
# expired (possibly poisoned) leaf. Widening only ever promotes, so the
# effective mode can demote but never return.
abstract type RollMode end
struct RefoldMode <: RollMode end
struct RunningMode <: RollMode end
struct TreeMode <: RollMode end

candidatemode(protos::Tuple) =
    all(p -> p isa GroupSummarizer, protos) ? RunningMode() :
    all(p -> p isa MonoidSummarizer, protos) ? TreeMode() : RefoldMode()

effectivemode(::RefoldMode, stateprotos::Tuple) = RefoldMode()
effectivemode(::TreeMode, stateprotos::Tuple) = TreeMode()
effectivemode(::RunningMode, stateprotos::Tuple) =
    all(isinvertible, stateprotos) ? RunningMode() : TreeMode()

struct RollingConfig{KN,LB<:Tuple,P<:Tuple,O,M<:RollMode}
    windownames::Tuple{Vararg{Symbol}}
    lookbacks::LB
    keycols::Vector{Symbol}
    keynames::Val{KN}
    protos::P
    outs::Val{O}
    prefixednames::Vector{Symbol}
    mode::M
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). The dynamically typed fields are per-chunk setup state;
# everything per-row sits behind the rollsegment! function barrier.
mutable struct RollingState
    schunks::Any       # summarized chunk iterator
    sstate::Any        # its iteration state
    sstarted::Bool
    sdone::Bool        # summarized stream exhausted
    snt::Any           # current summarized column table (nothing until pulled)
    spos::Int          # index of the next unadmitted summarized row in snt
    stypes::Union{Nothing,NamedTuple}  # promotion of summarized schemas seen
    stateprotos::Any   # state tuple template, fresh-cloned per row per window
    buffer::Any        # Vector{R}: admitted rows, time-ordered (refold/running)
    bufhead::Int       # first buffered row inside some window (refold mode)
    valtype::Any       # output value NamedTuple type (summary ∪ empty values)
    emptyrow::Any      # emptyvalues converted to valtype
    vals::Any          # per-window value vectors for the chunk in progress
    mode::Any          # effective RollMode for the realized state types
    winheads::Any      # Vector{Int}: per-window eviction head (running mode)
    rgroups::Any       # NTuple{W,Dict{K,RunningGroup{S}}} (running mode)
    trees::Any         # Dict{K,SegTree{S,R,T}} (tree mode; owns its rows)
    passthrough::Bool  # the summarized stream produced no chunks at all
    checked::Bool      # augmented-side name/key validation done
    RollingState(schunks) = new(schunks, nothing, false, false, nothing, 1,
                                nothing, nothing, nothing, 1, nothing, nothing,
                                nothing, RefoldMode(), nothing, nothing,
                                nothing, false, false)
end

# One key's running window state: the states with every in-window row folded
# in, and how many rows that is — zero-row groups are deleted rather than
# kept, so presence in the dict implies live >= 1 and an absent key emits
# the empty row, exactly as the re-fold path's `seen` flag decides it.
mutable struct RunningGroup{S<:Tuple}
    states::S
    live::Int
end

# Pull the next summarized chunk (type-unstable, once per summarized chunk):
# create or widen the state prototypes and buffer when the promoted schema
# moves. States are folded fresh per output row, so widening rebuilds only
# the prototypes — there is no accumulated state to carry — but the buffered
# rows and any half-filled value vectors convert along.
function pullsummarized!(rs::RollingState, cfg::RollingConfig)
    next = rs.sstarted ? iterate(rs.schunks, rs.sstate) : iterate(rs.schunks)
    rs.sstarted = true
    if next === nothing
        rs.sdone = true
        rs.snt === nothing && (rs.passthrough = true)
        return nothing
    end
    chunk, rs.sstate = next
    if rs.stypes === nothing
        for k in cfg.keycols
            String(k) in names(chunk) || throw(ArgumentError(
                "addrollingcolumns key column $k not found in the summarized input"))
        end
    end
    types = promotetypes(rs.stypes, chunktypes(chunk))
    widened = rs.stypes !== nothing && types != rs.stypes
    rs.stypes = types
    rs.snt = Tables.columntable(chunk)
    rs.spos = 1
    if rs.stateprotos === nothing
        rs.stateprotos = newstates(cfg.protos, types)
        rs.buffer = storerowtype(types)[]
        rs.bufhead = 1
        setvaltype!(rs, cfg)
        setupmode!(rs, cfg, types)
    elseif widened
        oldmode = rs.mode
        rs.stateprotos = widenstates(rs.stateprotos, types)
        rs.buffer = convert(Vector{storerowtype(types)}, rs.buffer)
        setvaltype!(rs, cfg)
        widenmode!(rs, cfg, types, oldmode)
    end
    return nothing
end

# Build the effective mode's structures for the first realized state types
# (type-unstable, once per run). Refold needs nothing beyond the buffer.
function setupmode!(rs::RollingState, cfg::RollingConfig, types::NamedTuple)
    rs.mode = effectivemode(cfg.mode, rs.stateprotos)
    if rs.mode isa RunningMode
        rs.winheads = ones(Int, length(cfg.lookbacks))
        rs.rgroups = newrgroups(rs.stateprotos,
                                storekeytype(types, cfg.keynames),
                                length(cfg.lookbacks))
    elseif rs.mode isa TreeMode
        rs.trees = newtrees(rs.stateprotos, storekeytype(types, cfg.keynames),
                            storerowtype(types), types.time)
    end
    return nothing
end

newrgroups(stateprotos::S, ::Type{K}, w::Int) where {S<:Tuple,K} =
    ntuple(_ -> Dict{K,RunningGroup{S}}(), w)
newtrees(stateprotos::S, ::Type{K}, ::Type{R}, ::Type{T}) where {S<:Tuple,K,R,T} =
    Dict{K,SegTree{S,R,T}}()

# Rebuild the incremental structures after a widening (type-unstable, rare):
# always from the live rows, the simple choice that is correct for every
# transition — including the running -> tree demotion when the widening let
# missing into an accumulator, after which the buffer goes unused.
function widenmode!(rs::RollingState, cfg::RollingConfig, types::NamedTuple,
                    oldmode::RollMode)
    rs.mode = effectivemode(cfg.mode, rs.stateprotos)
    K = storekeytype(types, cfg.keynames)
    if rs.mode isa RunningMode
        rs.rgroups = ntuple(length(cfg.lookbacks)) do w
            replaygroups!(Dict{K,RunningGroup{typeof(rs.stateprotos)}}(),
                          rs.buffer, rs.winheads[w], rs.stateprotos,
                          cfg.keynames)
        end
    elseif rs.mode isa TreeMode
        trees = newtrees(rs.stateprotos, K, storerowtype(types), types.time)
        if oldmode isa RunningMode
            replaytrees!(trees, rs.buffer, minimum(rs.winheads),
                         rs.stateprotos, cfg.keynames)
            empty!(rs.buffer)
            rs.rgroups = nothing
            rs.winheads = nothing
        else
            for (_, old) in rs.trees
                replaytrees!(trees, old.rows, old.head, rs.stateprotos,
                             cfg.keynames)
            end
        end
        rs.trees = trees
    end
    return nothing
end

# Fold the live rows (from head on) back into fresh per-key running groups;
# a function barrier so the per-row work is concretely typed.
function replaygroups!(d::Dict{K,RunningGroup{S}}, buffer::Vector, head::Int,
                       stateprotos::S, keynames::Val{KN}) where {K,S<:Tuple,KN}
    for j in head:length(buffer)
        row = @inbounds buffer[j]
        g = get!(() -> RunningGroup(map(fresh, stateprotos), 0), d,
                 keyvalues(row, keynames))
        foreach(st -> update!(st, row), g.states)
        g.live += 1
    end
    return d
end

# Push the live rows (from head on) into per-key trees, converting each to
# the trees' row type; a function barrier like replaygroups!.
function replaytrees!(trees::Dict{K,SegTree{S,R,T}}, rows::Vector, head::Int,
                      stateprotos::S,
                      keynames::Val{KN}) where {K,S<:Tuple,R,T,KN}
    for j in head:length(rows)
        row = convert(R, @inbounds rows[j])
        tr = get!(() -> newsegtree(stateprotos, R, T), trees,
                  keyvalues(row, keynames))
        treepush!(tr, stateprotos, row)
    end
    return trees
end

# The output value type: the summary values' NamedTuple type promoted
# field-wise with the empty values' (an empty window emits the latter), so
# e.g. Min over an Int column gives Union{Missing, Int}. promote_op can in
# principle fail to concretize, hence the Union fallback.
function setvaltype!(rs::RollingState, cfg::RollingConfig)
    VT = valuetype(typeof(rs.stateprotos), cfg.outs)
    e = emptyvalues(cfg.protos, cfg.outs)
    E = typeof(e)
    V = if VT <: NamedTuple && isconcretetype(VT) && fieldnames(VT) == keys(e)
        NamedTuple{keys(e),
                   Tuple{ntuple(i -> promote_type(fieldtype(VT, i),
                                                  fieldtype(E, i)),
                                fieldcount(E))...}}
    else
        Union{VT,E}
    end
    rs.valtype = V
    rs.emptyrow = convert(V, e)
    rs.vals === nothing ||
        (rs.vals = map(v -> convert(Vector{V}, v), rs.vals))
    return nothing
end

function rollchunk!(rs::RollingState, cfg::RollingConfig, c::DataFrame)
    if !rs.checked
        for k in cfg.keycols
            String(k) in names(c) || throw(ArgumentError(
                "addrollingcolumns key column $k not found in the augmented input"))
        end
        for pn in cfg.prefixednames
            String(pn) in names(c) && throw(ArgumentError(
                "rolling summary column $pn collides with an existing column"))
        end
        rs.checked = true
    end
    rs.snt === nothing && !rs.sdone && pullsummarized!(rs, cfg)
    rs.passthrough && return assembleempty(cfg, c)
    lnt = Tables.columntable(c)
    # Pre-filled with the empty row so a mid-chunk widen never converts an
    # undef slot.
    rs.vals = map(_ -> newvals(rs.valtype, rs.emptyrow, nrow(c)),
                  cfg.lookbacks)
    i = 1
    # The mode is re-read each pass: a widening inside pullsummarized! can
    # demote running to tree mid-chunk.
    while true
        if rs.mode isa RunningMode
            i, rs.spos, needpull =
                rollsegmentrunning!(rs.vals, rs.buffer, rs.winheads, lnt, i,
                                    rs.snt, rs.spos, rs.sdone, cfg.lookbacks,
                                    cfg.keynames, rs.stateprotos, rs.rgroups,
                                    cfg.outs, rs.emptyrow)
        elseif rs.mode isa TreeMode
            i, rs.spos, needpull =
                rollsegmenttree!(rs.vals, rs.trees, lnt, i, rs.snt, rs.spos,
                                 rs.sdone, cfg.lookbacks, cfg.keynames,
                                 rs.stateprotos, cfg.outs, rs.emptyrow)
        else
            i, rs.spos, rs.bufhead, needpull =
                rollsegment!(rs.vals, rs.buffer, rs.bufhead, lnt, i, rs.snt,
                             rs.spos, rs.sdone, cfg.lookbacks, cfg.keynames,
                             rs.stateprotos, cfg.outs, rs.emptyrow)
        end
        needpull || break
        pullsummarized!(rs, cfg)
    end
    if rs.mode isa RunningMode
        compactrunning!(rs.buffer, rs.winheads)
    elseif rs.mode isa RefoldMode
        rs.bufhead = compact!(rs.buffer, rs.bufhead)
    end
    return assemble(cfg, rs, c)
end

newvals(::Type{V}, emptyrow, n::Int) where {V} =
    fill!(Vector{V}(undef, n), emptyrow)

# Dead rows accumulate at the front of the buffer as the head advances;
# dropping them only when they dominate keeps the cost amortized O(1) per
# admitted row.
function compact!(buffer::Vector, head::Int)
    dead = head - 1
    if dead >= 64 && 2 * dead >= length(buffer)
        deleteat!(buffer, 1:dead)
        return 1
    end
    return head
end

# The running-mode analogue: a row is dead once the laggiest window's head
# has passed it, and the per-window heads shift down with the drop.
function compactrunning!(buffer::Vector, winheads::Vector{Int})
    dead = minimum(winheads) - 1
    if dead >= 64 && 2 * dead >= length(buffer)
        deleteat!(buffer, 1:dead)
        winheads .-= dead
    end
    return nothing
end

# --- folding kernel --------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access. Processes augmented rows from index i, admitting
# summarized rows from snt starting at spos into the buffer. Returns
# (i, spos, head, needpull): needpull means the current summarized chunk is
# consumed but the stream may still hold rows with time <= row i's time —
# the driver must pull the next summarized chunk before row i can be folded.
function rollsegment!(vals::Tuple, buffer::Vector{R}, head::Int,
                      lnt::NamedTuple, i::Int, snt::NamedTuple, spos::Int,
                      sdone::Bool, lookbacks::Tuple, keynames::Val{KN},
                      stateprotos::S, outs::Val,
                      emptyrow) where {R,KN,S<:Tuple}
    n = length(lnt.time)
    slen = length(snt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        # Admit summarized rows not after t — equal times included, so every
        # row tied at t is buffered before any row at t is folded.
        while spos <= slen && @inbounds(snt.time[spos]) <= t
            push!(buffer, rowat(R, snt, spos))
            spos += 1
        end
        spos > slen && !sdone && return (i, spos, head, true)
        # Times are non-decreasing, so a row outside every window now is
        # outside forever.
        while head <= length(buffer) &&
                outsideall(t - @inbounds(buffer[head]).time, lookbacks...)
            head += 1
        end
        foldwindows!(vals, i, t, buffer, head, keyat(lnt, i, keynames),
                     keynames, stateprotos, outs, emptyrow, lookbacks...)
        i += 1
    end
    return (i, spos, head, false)
end

@inline outsideall(d) = true
@inline outsideall(d, lb, rest...) = d > lb && outsideall(d, rest...)

# One window per call, peeling the value vectors and look-backs in step
# (vararg style, so inference tracks the heterogeneous look-back types).
# Fresh states per window per row: accumulate-only states cannot evict, and
# they are small mutable structs, so re-folding the window is the simple
# correct baseline. `seen` guards value: Min/Max/First/Last leave their
# value field undefined until a row is folded in.
@inline foldwindows!(::Tuple{}, i, t, buffer, head, k, keynames, stateprotos,
                     outs, emptyrow) = nothing
@inline function foldwindows!(vals::Tuple, i::Int, t, buffer::Vector,
                              head::Int, k::NamedTuple, keynames::Val,
                              stateprotos::Tuple, outs::Val, emptyrow,
                              lb, rest...)
    states = map(fresh, stateprotos)
    seen = false
    for j in head:length(buffer)
        s = @inbounds buffer[j]
        t - s.time <= lb || continue
        isequal(keyvalues(s, keynames), k) || continue
        foreach(st -> update!(st, s), states)
        seen = true
    end
    v = first(vals)
    @inbounds v[i] = seen ? summaryvalues(states, outs) : emptyrow
    return foldwindows!(Base.tail(vals), i, t, buffer, head, k, keynames,
                        stateprotos, outs, emptyrow, rest...)
end

# --- running (group) kernel ------------------------------------------------
#
# The same admit/evict/emit protocol as rollsegment!, but incremental: every
# admitted row is update!d into its key's running group in every window, and
# each window's eviction head downdate!s rows as they age out — O(1)
# amortized per row per window. The per-window dicts and value vectors are
# homogeneous tuples (indexing them by an Int is type-stable); only the
# heterogeneous look-backs peel vararg-style, carrying the window index w
# alongside.
function rollsegmentrunning!(vals::Tuple, buffer::Vector{R},
                             winheads::Vector{Int}, lnt::NamedTuple, i::Int,
                             snt::NamedTuple, spos::Int, sdone::Bool,
                             lookbacks::Tuple, keynames::Val{KN},
                             stateprotos::S, rgroups::G, outs::Val,
                             emptyrow) where {R,KN,S<:Tuple,G<:Tuple}
    n = length(lnt.time)
    slen = length(snt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        while spos <= slen && @inbounds(snt.time[spos]) <= t
            row = rowat(R, snt, spos)
            push!(buffer, row)
            # A row already outside a short window still enters its group;
            # the eviction sweep below downdates it right back out, keeping
            # the invariant that window w's groups hold exactly the rows in
            # buffer[winheads[w]:end], per key.
            admitrunning!(row, keynames, stateprotos, rgroups)
            spos += 1
        end
        spos > slen && !sdone && return (i, spos, true)
        evictrunning!(t, buffer, winheads, keynames, rgroups, 1, lookbacks...)
        k = keyat(lnt, i, keynames)
        for w in 1:length(vals)
            g = get(rgroups[w], k, nothing)
            v = vals[w]
            @inbounds v[i] = g === nothing ? emptyrow :
                summaryvalues(g.states, outs)
        end
        i += 1
    end
    return (i, spos, false)
end

@inline function admitrunning!(row, keynames::Val, stateprotos::Tuple,
                               rgroups::Tuple)
    for w in 1:length(rgroups)
        g = get!(() -> RunningGroup(map(fresh, stateprotos), 0), rgroups[w],
                 keyvalues(row, keynames))
        foreach(st -> update!(st, row), g.states)
        g.live += 1
    end
    return nothing
end

# One window per recursion step, peeling the look-backs with the window
# index in tow. Deleting a group when its last row leaves keeps the dicts
# bounded by the keys currently in some window, and makes "absent key" mean
# "empty window" for the emission above.
@inline evictrunning!(t, buffer::Vector, winheads::Vector{Int}, keynames::Val,
                      rgroups::Tuple, w::Int) = nothing
@inline function evictrunning!(t, buffer::Vector, winheads::Vector{Int},
                               keynames::Val, rgroups::Tuple, w::Int, lb,
                               rest...)
    d = rgroups[w]
    head = @inbounds winheads[w]
    while head <= length(buffer) && t - @inbounds(buffer[head]).time > lb
        row = @inbounds buffer[head]
        k = keyvalues(row, keynames)
        g = d[k]
        foreach(st -> downdate!(st, row), g.states)
        g.live -= 1
        g.live == 0 && delete!(d, k)
        head += 1
    end
    @inbounds winheads[w] = head
    return evictrunning!(t, buffer, winheads, keynames, rgroups, w + 1,
                         rest...)
end

# --- tree (monoid) kernel --------------------------------------------------
#
# The same protocol again, over per-key segment trees that own their rows —
# the shared buffer goes unused. Each admitted row appends to its key's
# tree; each output row binary-searches its window's start per look-back and
# folds the suffix from O(log n) partial combinations. The row minimum of
# the window starts then advances the tree's head: times are non-decreasing,
# so a row older than every window is expired for good, and the tree drops
# the prefix at its next rebuild.
function rollsegmenttree!(vals::Tuple, trees::Dict{K,SegTree{S,R,T}},
                          lnt::NamedTuple, i::Int, snt::NamedTuple, spos::Int,
                          sdone::Bool, lookbacks::Tuple, keynames::Val{KN},
                          stateprotos::S, outs::Val,
                          emptyrow) where {K,S<:Tuple,R,T,KN}
    n = length(lnt.time)
    slen = length(snt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        while spos <= slen && @inbounds(snt.time[spos]) <= t
            row = rowat(R, snt, spos)
            tr = get!(() -> newsegtree(stateprotos, R, T), trees,
                      keyvalues(row, keynames))
            treepush!(tr, stateprotos, row)
            spos += 1
        end
        spos > slen && !sdone && return (i, spos, true)
        k = keyat(lnt, i, keynames)
        tr = get(trees, k, nothing)
        if tr === nothing
            for w in 1:length(vals)
                v = vals[w]
                @inbounds v[i] = emptyrow
            end
        else
            minlo = emittree!(vals, i, t, tr, stateprotos, outs, emptyrow,
                              length(tr.rows) + 1, 1, lookbacks...)
            tr.head = max(tr.head, minlo)
        end
        i += 1
    end
    return (i, spos, false)
end

@inline emittree!(vals::Tuple, i::Int, t, tr::SegTree, stateprotos::Tuple,
                  outs::Val, emptyrow, minlo::Int, w::Int) = minlo
@inline function emittree!(vals::Tuple, i::Int, t, tr::SegTree,
                           stateprotos::Tuple, outs::Val, emptyrow,
                           minlo::Int, w::Int, lb, rest...)
    lo = windowstart(tr.times, tr.head, t, lb)
    hi = length(tr.rows)
    v = vals[w]
    @inbounds v[i] = lo > hi ? emptyrow :
        summaryvalues(treequery(tr, stateprotos, lo, hi), outs)
    return emittree!(vals, i, t, tr, stateprotos, outs, emptyrow,
                     min(minlo, lo), w + 1, rest...)
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
