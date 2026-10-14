# The interval-summarization transform. The data stream `p` drives a chunkmap
# and pulls the clock's boundaries on demand, as asofjoin pulls its right
# stream. It is the summarizecycles fold, closing when a clock boundary is
# crossed rather than when the timestamp changes. The declared-key path, which
# keyless output takes over the one empty key, also emits a row for every
# empty interval, making a regular grid.
# Type-unstable setup (pulling clock chunks, building or widening states) runs
# once per chunk; the folding kernels take a concretely typed boundary vector.

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
    (; keycols, protos, terms, ks, keynames, outs) =
        summaryspec(summarizers, key, "intervalize"; keyset, grid = true)
    return pipetransform() do p, ctx
        T = timetype(ctx)
        st = IntervalizeState{T}(IntervalCursor{T}(clk.run(ctx)), T[], 2,
            SummaryFold(terms, "intervalize"), false)
        step =
            ks === nothing ?
            (c -> intervalchunkgrouped!(st, protos, keycols, keynames, outs,
                closelast, c)) :
            (c -> intervalchunkdense!(st, protos, ks, keycols, keynames, outs,
                closelast, c))
        flush =
            ks === nothing ?
            (() -> intervalflushgrouped!(st, ctx.stop, closelast, outs)) :
            (() -> intervalflushdense!(st, protos, ks, ctx.stop, closelast, outs))
        return chunkmap(step, p.run(ctx); flush = flush)
    end
end
intervalize(p::CausalPipeline, clk::CausalPipeline, summarizers; kwargs...) =
    intervalize(clk, summarizers; kwargs...)(p)

# Per-run state. `bounds` holds the pulled boundaries not yet behind the
# current interval, which ends at `bounds[bi]` and begins at `bounds[bi-1]`.
# The SummaryFold's untyped fields are per-chunk setup; per-row work sits in
# the fold kernels.
mutable struct IntervalizeState{T}
    cur::IntervalCursor{T}
    bounds::Vector{T}
    bi::Int          # index of the current interval's end in `bounds`
    fold::SummaryFold
    checked::Bool    # key columns validated against the input schema
end

# Drop boundaries before the current interval's begin, keeping the buffer
# bounded. Afterwards the begin is at index 1 and the end at index 2.
function trimbounds!(st::IntervalizeState)
    st.bi > 2 || return nothing
    deleteat!(st.bounds, 1:(st.bi-2))
    st.bi = 2
    return nothing
end

# The no-data grid's times: the end of every complete interval, then `stop`
# under closelast.
gridtimes(bounds::Vector, stop, closelast::Bool) =
    closelast ? [bounds[2:end]; stop] : bounds[2:end]

# --- keyed (sparse) path ---------------------------------------------------

function intervalchunkgrouped!(st::IntervalizeState, protos::Tuple,
    keycols::Vector{Symbol}, keynames::Val, outs::Val,
    closelast::Bool, c::DataFrame)
    if !st.checked
        checkkeycolumns(keycols, c, "intervalize")
        st.checked = true
    end
    nt = preparechunk!(st.fold, protos, true, keynames, c)
    pullpast!(st.bounds, st.cur, last(nt.time))
    rows, st.bi = foldintervalsgrouped!(st.fold.groups, st.fold.stateprotos, nt,
        st.bounds, st.bi, keynames, closelast, outs)
    trimbounds!(st)
    return rowsframe(rows)
end

# The keyed analogue: closing an interval emits a row per present key, sorted,
# and empties the groups (closecycle!), so an empty interval emits nothing.
function foldintervalsgrouped!(groups::GroupTable{K,S}, stateprotos::S,
    nt::NamedTuple, bounds::Vector{T}, bi::Int, keynames::Val{KN},
    closelast::Bool, outs::Val) where {K,S,T,KN}
    rows = rowtype(T, K, S, outs)[]
    nb = length(bounds)
    nb == 0 && return rows, bi
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue
        while bi <= nb && t >= @inbounds(bounds[bi])
            closecycle!(rows, groups, @inbounds(bounds[bi]), outs)
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
    pullpast!(st.bounds, st.cur, nothing)
    isempty(st.bounds) && return nothing
    st.fold.groups === nothing && return nothing   # no data ever: emit nothing
    RT = rowtype(T, keytype(st.fold.groups), valtype(st.fold.groups), outs)
    rows = flushintervalsgrouped!(st.fold.groups, st.bounds, st.bi, RT, stop,
        closelast, outs)
    return rowsframe(rows)
end

# The current interval closes at its end, emitting its present keys; if data
# reached the trailing region instead, closelast closes it at stop. Later
# intervals are empty and emit nothing.
function flushintervalsgrouped!(groups::GroupTable{K,S}, bounds::Vector{T},
    bi::Int, ::Type{RT}, stop, closelast::Bool, outs::Val) where {K,S,T,RT}
    rows = RT[]
    if bi <= length(bounds)
        closecycle!(rows, groups, @inbounds(bounds[bi]), outs)
    elseif closelast
        closecycle!(rows, groups, stop, outs)
    end
    return rows
end

# --- declared-key (dense) path, keyless included --------------------------

function intervalchunkdense!(st::IntervalizeState{T}, protos::Tuple, ks::KeySet,
    keycols::Vector{Symbol}, keynames::Val, outs::Val, closelast::Bool,
    c::DataFrame) where {T}
    if !st.checked
        checkkeycolumns(keycols, c, "intervalize")
        st.checked = true
    end
    nt = preparechunk!(st.fold, protos, true, keynames, c; keyset = ks)
    pullpast!(st.bounds, st.cur, last(nt.time))
    RT, emptyrow = densetypes(T, st.fold.stateprotos, protos, ks, outs)
    rows = RT[]
    st.bi = foldintervalsdense!(rows, st.fold.groups, ks, nt, st.bounds, st.bi,
        keynames, closelast, outs, emptyrow)
    trimbounds!(st)
    return rowsframe(rows)
end

# The grid fold over declared-key slots: every crossed boundary closes all of
# them (closedense!), so an empty interval still emits a row per key. Rows that
# won't be emitted are skipped before their key is looked up. Returns `bi`.
function foldintervalsdense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet,
    nt::NamedTuple, bounds::Vector{T}, bi::Int, keynames::Val,
    closelast::Bool, outs::Val, emptyrow) where {RT,T}
    nb = length(bounds)
    nb == 0 && return bi
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue
        while bi <= nb && t >= @inbounds(bounds[bi])
            closedense!(rows, dg, ks, @inbounds(bounds[bi]), outs, emptyrow)
            bi += 1
        end
        (bi > nb && !closelast) && continue
        densefold!(dg, ks, row, keynames)
    end
    return bi
end

function intervalflushdense!(st::IntervalizeState{T}, protos::Tuple,
    ks::KeySet, stop::T, closelast::Bool, outs::Val) where {T}
    pullpast!(st.bounds, st.cur, nothing)
    isempty(st.bounds) && return nothing
    st.fold.stateprotos === nothing &&
        return emptygrid(gridtimes(st.bounds, stop, closelast), ks.keys,
            emptyvalues(protos, outs))
    RT, emptyrow = densetypes(T, st.fold.stateprotos, protos, ks, outs)
    rows = flushintervalsdense!(RT[], st.fold.groups, ks, st.bounds, st.bi,
        stop, closelast, outs, emptyrow)
    return rowsframe(rows)
end

# Drain the remaining grid: the current interval, every later complete one
# (empty), then the trailing partial at stop when closelast.
function flushintervalsdense!(rows::Vector{RT}, dg::DenseGroups, ks::KeySet,
    bounds::Vector{T}, bi::Int, stop::T, closelast::Bool, outs::Val,
    emptyrow) where {RT,T}
    nb = length(bounds)
    while bi <= nb
        closedense!(rows, dg, ks, @inbounds(bounds[bi]), outs, emptyrow)
        bi += 1
    end
    closelast && closedense!(rows, dg, ks, stop, outs, emptyrow)
    return rows
end
