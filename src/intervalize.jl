# The interval-summarization transform. A clock pipeline supplies the interval
# boundaries; the data stream `p` drives a chunkmap, and the clock is pulled on
# demand from inside the driver (like asofjoin's right stream and
# addrollingcolumns' summarized stream). It is the summarizecycles fold with
# the close trigger changed from "the timestamp changed" to "a clock boundary
# was crossed", plus — on the keyless path — an emitted row for every interval,
# empty ones included, to make the output a regular grid. Per the summarize.jl
# conventions, the type-unstable setup (pulling clock chunks, building/widening
# states) happens once per chunk and the folding kernels take a concretely
# typed boundary vector, so the per-row work stays dispatch-free.

"""
    intervalize(clock, summarizers; key = nothing, keyset = nothing,
                closelast = false) -> (CausalPipeline -> CausalPipeline)
    intervalize(p::CausalPipeline, clock, summarizers; ...) -> CausalPipeline

A transform summarizing the input over the intervals between consecutive clock
times `b₀ < b₁ < …`. The rows in each `[bₖ, bₖ₊₁)` are summarized and emitted
at `bₖ₊₁`, with columns `time`, the key columns, then the summaries; the input
columns are dropped. Rows before `b₀` are dropped.

# Arguments
- `clock`: a pipeline whose `:time` column gives the boundaries (other columns
  are ignored), such as [`clock`](@ref). An empty clock gives no output.
- `summarizers`: a [`Summarizer`](@ref) or a collection of them.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`. Without a key every interval emits one row, an empty interval
  its summarizers' empty values (`count = 0`, `mean = missing`). With a key each
  interval emits one row per key present, sorted by key, and an empty interval
  emits nothing.
- `keyset = nothing`: the key values, declared up front (requires `key`): a
  collection of values for one key column, or of tuples or named tuples for
  several. Every interval then emits one row per declared key, in declared
  order, with empty values for keys without rows. Key columns take `keyset`'s
  types, and a row with an undeclared key is an `ArgumentError`.
- `closelast = false`: also emit the partial interval after the last boundary,
  at `stop`.

Empty values widen the output types (`Mean` gives `Union{Missing, Float64}`).
"""
function intervalize(clk::CausalPipeline, summarizers; key = nothing,
    keyset = nothing, closelast::Bool = false)
    keycols = keycolumns(key, "intervalize")
    protos, requested = prototypes(tosummarizers(summarizers), keycols, "intervalize")
    ks = tokeyset(keyset, keycols, "intervalize")
    keynames = Val(Tuple(keycols))
    outs = Val(requested)
    keyed = !isempty(keycols)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            T = timetype(ctx)
            st = IntervalizeState{T}(IntervalCursor{T}(clk.run(ctx)), T[], 2,
                false, SummaryFold(), false, false)
            step = function (c)
                if ks !== nothing
                    intervalstepdense!(st, protos, ks, keycols, keynames, outs,
                        closelast, c)
                elseif keyed
                    intervalstepgrouped!(st, protos, keycols, keynames, outs,
                        closelast, c)
                else
                    intervalstep!(st, protos, keynames, outs, closelast, c)
                end
            end
            flush =
                ks !== nothing ?
                (() -> intervalflushdense!(st, protos, ks, ctx.stop, closelast,
                    outs)) :
                keyed ?
                (() -> intervalflushgrouped!(st, ctx.stop, closelast, outs)) :
                (() -> intervalflush!(st, protos, ctx.stop, closelast, outs))
            return chunkmap(step, p.run(ctx); flush = flush)
        end
    end
end
intervalize(p::CausalPipeline, clk::CausalPipeline, summarizers; kwargs...) =
    intervalize(clk, summarizers; kwargs...)(p)

# The clock cursor: one boundary time at a time over clk.run(ctx), refilling
# from clock chunks lazily. The type-unstable pull (the chunks iterator's
# element type is opaque, and a DataFrame column access is dynamic, as in
# asofjoin's right stream) is confined here and to the driver; the per-boundary
# indexing of the concrete Vector{T} is typed. Clock order is trusted to the
# chunk protocol, exactly as asofjoin trusts its right stream.
mutable struct IntervalCursor{T}
    const chunks::PullCursor
    times::Vector{T}
    pos::Int
end
IntervalCursor{T}(chunks) where {T} = IntervalCursor{T}(PullCursor(chunks), T[], 1)

function nextboundary!(cur::IntervalCursor{T}) where {T}
    while cur.pos > length(cur.times)
        chunk = pull!(cur.chunks)
        chunk === nothing && return nothing
        cur.times = convert(Vector{T}, chunk.time)
        cur.pos = 1
    end
    b = @inbounds cur.times[cur.pos]
    cur.pos += 1
    return b
end

# Per-run mutable state, in fields rather than reassigned closure captures
# (those get boxed). `bounds` holds the pulled boundaries not yet fully behind
# the current interval, whose end is `bounds[bi]` (its begin `bounds[bi-1]`);
# they carry across chunk boundaries. The SummaryFold's dynamically typed
# fields are per-chunk setup state, and everything per-row sits behind the fold
# kernels' concretely typed boundary vector.
mutable struct IntervalizeState{T}
    cur::IntervalCursor{T}
    bounds::Vector{T}
    bi::Int          # index of the current interval's end in `bounds`
    donebounds::Bool # the clock has been drained into `bounds`
    fold::SummaryFold
    folded::Bool     # keyless: the current interval has folded a row
    checked::Bool    # keyed: key columns validated against the input schema
end

# Pull boundaries until the last one is strictly past `tmax`, so every data row
# in the chunk (all `<= tmax`) has a known enclosing interval, or the clock is
# exhausted. Type-unstable (the clock pull), run once per chunk.
function fillbounds!(st::IntervalizeState{T}, tmax::T) where {T}
    while !st.donebounds && (isempty(st.bounds) || @inbounds(st.bounds[end]) <= tmax)
        b = nextboundary!(st.cur)
        b === nothing ? (st.donebounds = true) : push!(st.bounds, b)
    end
    return nothing
end

# Drop boundaries fully behind the current interval's begin (`bounds[bi-1]`),
# keeping the buffer bounded across chunks. After it, the begin is at index 1
# and the end at index 2.
function trimbounds!(st::IntervalizeState)
    st.bi > 2 || return nothing
    deleteat!(st.bounds, 1:(st.bi-2))
    st.bi = 2
    return nothing
end

# Pull every remaining boundary into `bounds` (type-unstable; only at flush).
function drainbounds!(st::IntervalizeState)
    while !st.donebounds
        b = nextboundary!(st.cur)
        b === nothing ? (st.donebounds = true) : push!(st.bounds, b)
    end
    return nothing
end

# The output row type of the keyless grid: `:time` prepended to the value type
# (which already promotes summary and empty values via promotedvaluetype).
_intervalrow(t, v) = merge((; time = t), v)
intervalrowtype(::Type{T}, ::Type{V}) where {T,V} =
    Base.promote_op(_intervalrow, T, V)

# One keyless grid row: the summary of the interval if it folded a row,
# otherwise the empty values, both landing in the promoted row type RT.
@inline function closeintervalrow(::Type{RT}, t, states::Tuple, folded::Bool,
    emptyrow, r::Val) where {RT}
    folded && return convert(RT, merge((; time = t), summaryvalues(states, r)))
    return convert(RT, merge((; time = t), emptyrow))
end

# --- keyless (grid) path ---------------------------------------------------

function intervalstep!(st::IntervalizeState, protos::Tuple, keynames::Val,
    outs::Val, closelast::Bool, c::DataFrame)
    nt = preparechunk!(st.fold, protos, false, keynames, c)
    fillbounds!(st, last(nt.time))
    rows, st.fold.states, st.bi, st.folded =
        foldintervals!(st.fold.states, protos, nt,
            st.bounds, st.bi, st.folded, closelast, outs)
    trimbounds!(st)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# Close every interval a row crosses (emitting an empty row for each interval
# with no data, so the grid stays regular), then fold the row into the current
# interval. `bounds[bi]` is the current interval's end; `bi > length(bounds)`
# is the trailing region (the clock is drained past `tmax`), whose rows are
# folded only when closelast will emit them. Returns the closed rows and the
# carried state.
function foldintervals!(states::S, protos::P, nt::NamedTuple,
    bounds::Vector{T}, bi::Int, folded::Bool, closelast::Bool,
    r::Val) where {S<:Tuple,P<:Tuple,T}
    V = promotedvaluetype(S, protos, r)
    emptyrow = convert(V, emptyvalues(protos, r))
    RT = intervalrowtype(T, V)
    rows = RT[]
    nb = length(bounds)
    nb == 0 && return rows, states, bi, folded    # empty clock: drop every row
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue        # before the first interval
        while bi <= nb && t >= @inbounds(bounds[bi])
            push!(
                rows,
                closeintervalrow(RT, @inbounds(bounds[bi]), states,
                    folded, emptyrow, r),
            )
            # closeintervalrow copied the values out, so the closed interval's
            # states are zeroed and reused rather than replaced.
            states = freshall!(states)
            folded = false
            bi += 1
        end
        (bi > nb && !closelast) && continue          # discarded trailing row
        updateall!(states, row)
        folded = true
    end
    return rows, states, bi, folded
end

function intervalflush!(st::IntervalizeState{T}, protos::Tuple, stop::T,
    closelast::Bool, outs::Val) where {T}
    drainbounds!(st)
    isempty(st.bounds) && return nothing           # empty clock
    # No data ever arrived, so no states were built: the grid is all empty
    # values, typed from the summarizer configs alone.
    st.fold.stateprotos === nothing &&
        return flushemptygrid!(st, protos, stop, closelast, outs)
    rows = flushintervals!(st.fold.states, protos,
        st.bounds, st.bi, st.folded, stop, closelast, outs)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# Drain the remaining grid: the current interval (with data if folded) and
# every remaining complete interval up to b_K (each empty), then the trailing
# partial at stop when closelast.
function flushintervals!(states::S, protos::P, bounds::Vector{T},
    bi::Int, folded::Bool, stop::T, closelast::Bool,
    r::Val) where {S<:Tuple,P<:Tuple,T}
    V = promotedvaluetype(S, protos, r)
    emptyrow = convert(V, emptyvalues(protos, r))
    RT = intervalrowtype(T, V)
    rows = RT[]
    nb = length(bounds)
    while bi <= nb
        push!(
            rows,
            closeintervalrow(RT, @inbounds(bounds[bi]), states, folded,
                emptyrow, r),
        )
        states = freshall!(states)
        folded = false
        bi += 1
    end
    closelast && push!(rows, closeintervalrow(RT, stop, states, folded, emptyrow, r))
    return rows
end

# The no-data grid: an empty row per complete interval `[bounds[k-1], bounds[k])`
# (and the trailing partial when closelast), with the value type from the
# configs alone.
function flushemptygrid!(st::IntervalizeState{T}, protos::Tuple, stop::T,
    closelast::Bool, r::Val) where {T}
    e = emptyvalues(protos, r)
    RT = intervalrowtype(T, typeof(e))
    rows = RT[]
    for k in 2:length(st.bounds)
        push!(rows, convert(RT, merge((; time = @inbounds(st.bounds[k])), e)))
    end
    closelast && !isempty(st.bounds) &&
        push!(rows, convert(RT, merge((; time = stop), e)))
    return isempty(rows) ? nothing : DataFrame(rows)
end

# --- keyed (sparse) path ---------------------------------------------------

function intervalstepgrouped!(st::IntervalizeState, protos::Tuple,
    keycols::Vector{Symbol}, keynames::Val, outs::Val,
    closelast::Bool, c::DataFrame)
    if !st.checked
        checkkeycolumns(keycols, c, "intervalize")
        st.checked = true
    end
    nt = preparechunk!(st.fold, protos, true, keynames, c)
    fillbounds!(st, last(nt.time))
    rows, st.bi = foldintervalsgrouped!(st.fold.groups, st.fold.stateprotos, nt,
        st.bounds, st.bi, keynames, closelast, outs)
    trimbounds!(st)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# The keyed analogue: closing an interval emits one row per present key
# (sorted) and empties the groups (closecycle!), so empty intervals emit
# nothing — the grid is sparse per key, as unseen keys cannot be emitted.
function foldintervalsgrouped!(groups::GroupTable{K,S}, stateprotos::S,
    nt::NamedTuple, bounds::Vector{T}, bi::Int, keynames::Val{KN},
    closelast::Bool, r::Val) where {K,S,T,KN}
    rows = rowtype(T, K, S, r)[]
    nb = length(bounds)
    nb == 0 && return rows, bi
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue
        while bi <= nb && t >= @inbounds(bounds[bi])
            closecycle!(rows, groups, @inbounds(bounds[bi]), r)
            bi += 1
        end
        (bi > nb && !closelast) && continue
        states = groupstates!(groups, keyvalues(row, keynames), stateprotos)
        updateall!(states, row)
    end
    return rows, bi
end

function intervalflushgrouped!(st::IntervalizeState{T}, stop::T, closelast::Bool,
    outs::Val) where {T}
    drainbounds!(st)
    isempty(st.bounds) && return nothing
    st.fold.groups === nothing && return nothing   # no data ever: emit nothing
    RT = rowtype(T, keytype(st.fold.groups), valtype(st.fold.groups), outs)
    rows = flushintervalsgrouped!(st.fold.groups, st.bounds, st.bi, RT, stop,
        closelast, outs)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# The current interval closes at its end, emitting its present keys; if data
# reached the trailing region instead (bi past the last boundary), closelast
# closes it at stop. Intervening complete intervals are empty, so they emit
# nothing (the groups are emptied by the first close).
function flushintervalsgrouped!(groups::GroupTable{K,S}, bounds::Vector{T},
    bi::Int, ::Type{RT}, stop, closelast::Bool, r::Val) where {K,S,T,RT}
    rows = RT[]
    if bi <= length(bounds)
        closecycle!(rows, groups, @inbounds(bounds[bi]), r)
    elseif closelast
        closecycle!(rows, groups, stop, r)
    end
    return rows
end

# --- declared-key (dense) path ---------------------------------------------

function intervalstepdense!(st::IntervalizeState{T}, protos::Tuple, ks::KeySet,
    keycols::Vector{Symbol}, keynames::Val, outs::Val, closelast::Bool,
    c::DataFrame) where {T}
    if !st.checked
        checkkeycolumns(keycols, c, "intervalize")
        st.checked = true
    end
    nt = preparechunk!(st.fold, protos, true, keynames, c; keyset = ks)
    fillbounds!(st, last(nt.time))
    RT, emptyrow = densetypes(T, st.fold.stateprotos, protos, ks, outs)
    rows = RT[]
    st.bi = foldintervalsdense!(rows, st.fold.groups, ks, nt, st.bounds, st.bi,
        keynames, closelast, outs, emptyrow)
    trimbounds!(st)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# The keyless grid's fold over declared-key slots: every crossed boundary
# closes all of them (closedense!), so an empty interval still emits a row per
# key. Rows before the first boundary, and trailing rows closelast will not
# emit, are skipped before their key is looked up. Returns the interval index.
function foldintervalsdense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet,
    nt::NamedTuple, bounds::Vector{T}, bi::Int, keynames::Val,
    closelast::Bool, r::Val, emptyrow) where {RT,T}
    nb = length(bounds)
    nb == 0 && return bi
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue
        while bi <= nb && t >= @inbounds(bounds[bi])
            closedense!(rows, dg, ks, @inbounds(bounds[bi]), r, emptyrow)
            bi += 1
        end
        (bi > nb && !closelast) && continue
        densefold!(dg, ks, row, keynames)
    end
    return bi
end

function intervalflushdense!(st::IntervalizeState{T}, protos::Tuple,
    ks::KeySet, stop::T, closelast::Bool, outs::Val) where {T}
    drainbounds!(st)
    isempty(st.bounds) && return nothing
    st.fold.stateprotos === nothing &&
        return flushemptygrid!(st, protos, ks, stop, closelast, outs)
    RT, emptyrow = densetypes(T, st.fold.stateprotos, protos, ks, outs)
    rows = flushintervalsdense!(RT[], st.fold.groups, ks, st.bounds, st.bi,
        stop, closelast, outs, emptyrow)
    return isempty(rows) ? nothing : DataFrame(rows)
end

# `flushintervals!` over the slots: the current interval, every remaining
# complete one (each empty), then the trailing partial at stop when closelast.
function flushintervalsdense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet,
    bounds::Vector{T}, bi::Int, stop::T, closelast::Bool, r::Val,
    emptyrow) where {RT,T}
    nb = length(bounds)
    while bi <= nb
        closedense!(rows, dg, ks, @inbounds(bounds[bi]), r, emptyrow)
        bi += 1
    end
    closelast && closedense!(rows, dg, ks, stop, r, emptyrow)
    return rows
end

# The no-data dense grid: an empty row per complete interval per declared key
# (and the trailing partial when closelast), typed from the configs alone.
function flushemptygrid!(st::IntervalizeState{T}, protos::Tuple,
    ks::KeySet{K}, stop::T, closelast::Bool, r::Val) where {T,K}
    e = emptyvalues(protos, r)
    RT = gridrowtype(T, K, typeof(e))
    rows = RT[]
    for i in 2:length(st.bounds), k in ks.keys
        push!(rows, convert(RT, gridrow(@inbounds(st.bounds[i]), k, e)))
    end
    if closelast
        for k in ks.keys
            push!(rows, convert(RT, gridrow(stop, k, e)))
        end
    end
    return isempty(rows) ? nothing : DataFrame(rows)
end
