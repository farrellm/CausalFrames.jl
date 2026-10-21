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
    (; keycols, terms, cfg) =
        summaryspec(summarizers, key, "intervalize"; keyset, grid = true)
    return pipetransform() do p, ctx
        T = timetype(ctx)
        st = IntervalizeState{T}(IntervalCursor{T}(clk.run(ctx)), T[], 2,
            SummaryFold(cfg, keycols, terms, "intervalize"), closelast)
        return chunkmap(c -> intervalchunk!(st, c), p.run(ctx);
            flush = () -> intervalflush!(st, ctx.stop))
    end
end
intervalize(p::CausalPipeline, clk::CausalPipeline, summarizers; kwargs...) =
    intervalize(clk, summarizers; kwargs...)(p)

# Per-run state. `bounds` holds the pulled boundaries not yet behind the
# current interval, which ends at `bounds[bi]` and begins at `bounds[bi-1]`.
# The SummaryFold's untyped fields (and its erased config) are per-chunk setup;
# per-row work sits in the fold kernels.
mutable struct IntervalizeState{T}
    cur::IntervalCursor{T}
    bounds::Vector{T}
    bi::Int          # index of the current interval's end in `bounds`
    const fold::SummaryFold
    const closelast::Bool
end

function intervalchunk!(st::IntervalizeState, c::DataFrame)
    (; protos, grouping, outs) = st.fold.cfg
    nt = preparechunk!(st.fold, c)
    pullpast!(st.bounds, st.cur, last(nt.time))
    rows, st.bi = foldintervals!(st.fold.groups, st.fold.stateprotos, nt,
        st.bounds, st.bi, grouping, st.closelast, protos, outs)
    trimbounds!(st)
    return rowsframe(rows)
end

# Drop boundaries before the current interval's begin, keeping the buffer
# bounded. Afterwards the begin is at index 1 and the end at index 2.
function trimbounds!(st::IntervalizeState)
    st.bi > 2 || return nothing
    deleteat!(st.bounds, 1:(st.bi-2))
    st.bi = 2
    return nothing
end

function intervalflush!(st::IntervalizeState{T}, stop::T) where {T}
    (; protos, grouping, outs) = st.fold.cfg
    (; closelast) = st
    pullpast!(st.bounds, st.cur, nothing)
    isempty(st.bounds) && return nothing           # empty clock
    if st.fold.groups === nothing
        # No data ever arrived, so no states exist: a dense grid is all empty
        # values, typed from the summarizers alone, and sparse output is empty.
        grouping isa KeySet || return nothing
        return emptygrid(gridtimes(st.bounds, stop, closelast), grouping.keys,
            emptyvalues(protos, outs))
    end
    return rowsframe(
        flushintervals!(st.fold.groups, st.fold.stateprotos,
            st.bounds, st.bi, stop, grouping, closelast, protos, outs),
    )
end

# The no-data grid's times: the end of every complete interval, then `stop`
# under closelast.
gridtimes(bounds::Vector, stop, closelast::Bool) =
    closelast ? [bounds[2:end]; stop] : bounds[2:end]

# --- kernels ---------------------------------------------------------------
#
# Close every interval a row crosses, then fold the row into the current
# interval. `bi > length(bounds)` is the trailing region after the last
# boundary, whose rows are folded only under closelast; rows before the first
# boundary, or of an empty clock, are dropped. Returns the closed rows and
# `bi`. Keyed, closing an interval emits a row per present key, sorted, and
# empties the groups (`closecycle!`), so an empty interval emits nothing;
# over declared keys (keyless included), it closes every slot (`closedense!`),
# so an empty interval still emits a row per key.
function foldintervals!(groups, stateprotos::Tuple, nt::NamedTuple,
    bounds::Vector{T}, bi::Int, grouping, closelast::Bool, protos::Tuple,
    outs::Val) where {T}
    rows, emptyrow = closerows(groups, T, stateprotos, protos, grouping, outs)
    nb = length(bounds)
    nb == 0 && return rows, bi
    for row in Tables.rows(nt)
        t = row.time
        t < @inbounds(bounds[1]) && continue
        while bi <= nb && t >= @inbounds(bounds[bi])
            closegroups!(rows, groups, grouping, @inbounds(bounds[bi]), outs,
                emptyrow)
            bi += 1
        end
        (bi > nb && !closelast) && continue
        foldrow!(groups, stateprotos, grouping, row)
    end
    return rows, bi
end

# Drain at the end of the stream. Keyed, the current interval closes at its
# end, emitting its present keys, or, if data reached the trailing region
# instead, closelast closes it at stop; later intervals are empty and emit
# nothing. Dense, the current interval and every later complete one (empty)
# close, then the trailing partial at stop under closelast.
function flushintervals!(gt::GroupTable, stateprotos::Tuple, bounds::Vector{T},
    bi::Int, stop::T, keynames::Val, closelast::Bool, protos::Tuple,
    outs::Val) where {T}
    rows, _ = closerows(gt, T, stateprotos, protos, keynames, outs)
    if bi <= length(bounds)
        closecycle!(rows, gt, @inbounds(bounds[bi]), outs)
    elseif closelast
        closecycle!(rows, gt, stop, outs)
    end
    return rows
end
function flushintervals!(dg::DenseGroups, stateprotos::Tuple, bounds::Vector{T},
    bi::Int, stop::T, ks::KeySet, closelast::Bool, protos::Tuple,
    outs::Val) where {T}
    rows, emptyrow = closerows(dg, T, stateprotos, protos, ks, outs)
    for i in bi:length(bounds)
        closedense!(rows, dg, ks, @inbounds(bounds[i]), outs, emptyrow)
    end
    closelast && closedense!(rows, dg, ks, stop, outs, emptyrow)
    return rows
end
