# The rolling-window summarization transform. The augmented stream drives a
# chunkmap; the summarized stream is pulled on demand from inside the step,
# its rows buffered until they leave every window. Summarizer states are
# accumulate-only (no removal), so each output row folds fresh states over
# the buffered window rows rather than evicting incrementally — the planned
# invertible ("group") summarizer subtype is the future O(1) refinement. Per
# the summarize.jl conventions, the type-unstable setup happens once per
# chunk and the folding kernel takes concretely typed arguments behind a
# function barrier.

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
    schunks::Any       # summarized chunk iterator
    sstate::Any        # its iteration state
    sstarted::Bool
    sdone::Bool        # summarized stream exhausted
    snt::Any           # current summarized column table (nothing until pulled)
    spos::Int          # index of the next unadmitted summarized row in snt
    stypes::Union{Nothing,NamedTuple}  # promotion of summarized schemas seen
    stateprotos::Any   # state tuple template, fresh-cloned per row per window
    buffer::Any        # Vector{R}: admitted summarized rows, time-ordered
    bufhead::Int       # index of the first buffered row inside some window
    valtype::Any       # output value NamedTuple type (summary ∪ empty values)
    emptyrow::Any      # emptyvalues converted to valtype
    vals::Any          # per-window value vectors for the chunk in progress
    passthrough::Bool  # the summarized stream produced no chunks at all
    checked::Bool      # augmented-side name/key validation done
    RollingState(schunks) = new(schunks, nothing, false, false, nothing, 1,
                                nothing, nothing, nothing, 1, nothing, nothing,
                                nothing, false, false)
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
    elseif widened
        rs.stateprotos = widenstates(rs.stateprotos, types)
        rs.buffer = convert(Vector{storerowtype(types)}, rs.buffer)
        setvaltype!(rs, cfg)
    end
    return nothing
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
    while true
        i, rs.spos, rs.bufhead, needpull =
            rollsegment!(rs.vals, rs.buffer, rs.bufhead, lnt, i, rs.snt,
                         rs.spos, rs.sdone, cfg.lookbacks, cfg.keynames,
                         rs.stateprotos, cfg.outs, rs.emptyrow)
        needpull || break
        pullsummarized!(rs, cfg)
    end
    rs.bufhead = compact!(rs.buffer, rs.bufhead)
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
