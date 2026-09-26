# Acausal operations. Everything here looks *forward* in time, breaking the
# causal invariant, so it lives in a submodule reached only through
# `using CausalFrames.Acausal` and is never re-exported. The forward join runs
# on asofjoin's engine (join.jl) and supplies only its direction: a store and
# kernel picked by dispatch on `Forward`, with the match direction, tie-break
# and context widening inverted.
module Acausal

using ..CausalFrames: CausalPipeline, Context, chunkmap, shiftchunk!,
    settimechunk!, SetTimeState, checktimespec, timetype, keycolumns,
    normprefix, rowat, keyat, JoinConfig, JoinState, joinchunk!
import ..CausalFrames: newstore, widenstore, segment!

export futurejoin, lead

"""
    futurejoin(right::CausalPipeline; key = nothing, tolerance = nothing,
               strict = false, leftprefix = nothing, rightprefix = nothing,
               righttime = nothing) -> (CausalPipeline -> CausalPipeline)
    futurejoin(left::CausalPipeline, right::CausalPipeline; ...) -> CausalPipeline

**Acausal.** The mirror of [`asofjoin`](@ref CausalFrames.asofjoin): joins each
left row to the earliest `right` row at or after its time. Every left row is
kept, with `right`'s non-time columns appended as `Union{Missing, T}`:
`missing` where no right row matches. Among right rows at the same time, the
first one wins. Available after `using CausalFrames.Acausal`.

# Arguments
- `right`: the pipeline to join from. If it produces no rows, left rows pass
  through unchanged (apart from `leftprefix`).

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present on both sides; a row matches only right rows with an
  equal key. Key columns appear once, from the left, never prefixed.
- `tolerance = nothing`: the maximum distance ahead, `righttime - time <=
  tolerance`; must be non-negative. The right pipeline then runs over
  `[start, stop + tolerance)`, so rows near `stop` can match later right rows;
  the time type must support addition. Without it, right runs over
  `[start, stop)` only.
- `strict = false`: match only right rows strictly after the left row.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique, so a self
  join needs a prefix.
- `righttime = nothing`: a name under which to keep the matched right row's
  time; by default it is dropped.

Right rows are buffered per key until a left row consumes or passes them, and
proving a key has no future match reads the rest of `right`, so memory is
O(right rows) in the worst case, against `asofjoin`'s O(keys).
"""
function futurejoin(right::CausalPipeline; key = nothing, tolerance = nothing,
    strict::Bool = false, leftprefix = nothing,
    rightprefix = nothing,
    righttime::Union{Nothing,Symbol} = nothing)
    keycols = keycolumns(key, "futurejoin")
    righttime === :time && throw(
        ArgumentError(
            "futurejoin righttime may not be :time; it would collide with the left time column",
        ),
    )
    righttime !== nothing && righttime in keycols &&
        throw(
            ArgumentError(
                "futurejoin righttime $(repr(righttime)) collides with a key column"),
        )
    lp = normprefix(leftprefix)
    rp = normprefix(rightprefix)
    return function (left::CausalPipeline)
        return CausalPipeline() do ctx::Context
            cfg = JoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                strict ? (>) : (>=), Forward(), lp, rp, righttime, "futurejoin")
            js = JoinState(right.run(futurecontext(ctx, tolerance)))
            return chunkmap(c -> joinchunk!(js, cfg, c), left.run(ctx))
        end
    end
end
futurejoin(left::CausalPipeline, right::CausalPipeline; kwargs...) =
    futurejoin(right; kwargs...)(left)

# The mirror of `widenstart`: a tolerance widens the right window forward, so
# lookahead past the window's end is covered. The guard rejects a negative
# tolerance, which the Context constructor would accept.
futurecontext(ctx::Context, ::Nothing) = ctx
function futurecontext(ctx::Context, tolerance)
    stop = ctx.stop + tolerance
    stop >= ctx.stop || throw(ArgumentError(
        "futurejoin tolerance must be non-negative, got $tolerance"))
    return Context(ctx.start, stop)
end

# The forward direction of the join engine: the same driver, with the store and
# kernel below picked by dispatch, and `cmp` being `>` or `>=`.
struct Forward end

# A per-key FIFO of buffered future right rows: append on pull, pop by
# advancing `head`, with amortized compaction so the backing vector stays
# bounded. `matches` holds copied rows, not indices, so compaction never
# invalidates a match.
mutable struct KeyBuffer{V}
    rows::Vector{V}
    head::Int
end
KeyBuffer{V}() where {V} = KeyBuffer{V}(V[], 1)

@inline bufempty(b::KeyBuffer) = b.head > length(b.rows)
@inline buffront(b::KeyBuffer) = @inbounds b.rows[b.head]
@inline bufpush!(b::KeyBuffer, v) = (push!(b.rows, v); nothing)
function bufpop!(b::KeyBuffer)
    b.head += 1
    if b.head > length(b.rows)
        empty!(b.rows)
        b.head = 1
    elseif 2 * (b.head - 1) > length(b.rows)
        b.rows = b.rows[b.head:end]
        b.head = 1
    end
    return nothing
end

# A Dict of mutable `KeyBuffer`s: a lookup answers with a pointer, so there is
# no `Union{Nothing,V}` to box. A widening may arrive mid-left-chunk, so the
# per-key buffers convert along with the matches buffer.
newstore(::Forward, ::Type{K}, ::Type{V}) where {K,V} = Dict{K,KeyBuffer{V}}()
widenstore(store::Dict{<:Any,<:KeyBuffer}, ::Type{K}, ::Type{V}) where {K,V} =
    Dict{K,KeyBuffer{V}}(
        convert(K, k) => KeyBuffer{V}(convert(Vector{V}, b.rows), b.head)
        for (k, b) in store)

# --- merge kernel ----------------------------------------------------------
#
# Called with concretely typed arguments. First buffers the whole right chunk
# by key (a forward join needs rows *ahead* of t), then processes left rows
# from index i. Returns (i, rpos, needpull): needpull means row i's key has no
# buffered future row but the right stream isn't exhausted, so the driver must
# pull the next right chunk before resolving row i.
segment!(store::Dict{K,KeyBuffer{V}}, matches::Vector{V}, found::Vector{Bool},
    lnt::NamedTuple, i::Int, rnt::NamedTuple, rpos::Int, rdone::Bool,
    keynames::Val, after, tolerance) where {K,V} =
    futuresegment!(matches, found, store, lnt, i, rnt, rpos, rdone, keynames,
        after, tolerance)

function futuresegment!(matches::Vector{V}, found::Vector{Bool},
    store::Dict{K,KeyBuffer{V}}, lnt::NamedTuple, i::Int, rnt::NamedTuple,
    rpos::Int, rdone::Bool, keynames::Val{KN}, after::A,
    tolerance) where {K,V,KN,A}
    rlen = length(rnt.time)
    while rpos <= rlen
        b = get!(() -> KeyBuffer{V}(), store, keyat(rnt, rpos, keynames))
        bufpush!(b, rowat(V, rnt, rpos))
        rpos += 1
    end
    n = length(lnt.time)
    while i <= n
        t = @inbounds lnt.time[i]
        b = get(store, keyat(lnt, i, keynames), nothing)
        # Drop buffered rows before t (at or before t, if strict); they can't
        # match this or any later left row.
        if b !== nothing
            while !bufempty(b) && !after(buffront(b).time, t)
                bufpop!(b)
            end
        end
        if b === nothing || bufempty(b)
            # No buffered future row for this key: pull and re-enter at i if
            # the right stream may hold one, else leave it missing.
            rdone || return (i, rpos, true)
        else
            # The front is the earliest right row at or after t (after, if
            # strict). A front beyond tolerance may still match a later left
            # row, so it is never evicted on tolerance.
            m = buffront(b)
            if tolerance === nothing || m.time - t <= tolerance
                @inbounds matches[i] = m
                @inbounds found[i] = true
            end
        end
        i += 1
    end
    return (i, rpos, false)
end

# --- lead: the acausal time shift -----------------------------------------

"""
    lead(offset) -> (CausalPipeline -> CausalPipeline)
    lead(p::CausalPipeline, offset) -> CausalPipeline

**Acausal.** The mirror of [`lag`](@ref CausalFrames.lag): moves every row
`offset` earlier (`time -> time - offset`), so the row at time `t` carries the
values the input had at `t + offset`. Only `:time` changes. The input runs over
`[start + offset, stop + offset)`, so the output fills the whole window.
Available after `using CausalFrames.Acausal`.

# Arguments
- `offset`: the shift, in a type that can be added to and subtracted from the
  time type. Must be non-negative, checked when the pipeline runs; `0` is the
  identity.
"""
function lead(offset)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> shiftchunk!(c, -offset), p.run(leadcontext(ctx, offset)))
        end
    end
end
lead(p::CausalPipeline, offset) = lead(offset)(p)

# The mirror of `lagcontext`: the window slides forward by `offset`, so the
# shift lands the output in [start, stop). The guard rejects a negative offset,
# which would silently act as a lag.
function leadcontext(ctx::Context, offset)
    start = ctx.start + offset
    start >= ctx.start ||
        throw(ArgumentError("lead offset must be non-negative, got $offset"))
    return Context(start, ctx.stop + offset)
end

# --- settime: the permissive time reassignment -----------------------------

"""
    CausalFrames.Acausal.settime(spec) -> (CausalPipeline -> CausalPipeline)
    CausalFrames.Acausal.settime(p::CausalPipeline, spec) -> CausalPipeline

**Acausal.** The permissive [`settime`](@ref CausalFrames.settime): the same
`spec`, conversion and clip to `[start, stop)`, but rows may move earlier. The
new time column must still be non-decreasing within and across chunks (an
`ArgumentError` otherwise).

Not exported even from `Acausal`, so that `using CausalFrames.Acausal` leaves
the causal `settime` unambiguous; call it as `CausalFrames.Acausal.settime`.

# Arguments
- `spec`: a `Symbol` or a function `row -> time`, as for `settime`.

The input is not widened, so a row at or after `stop` is never seen, even if
`spec` would move it into the window. For a constant shift, use
[`lead`](@ref), which does widen.
"""
function settime(spec)
    checktimespec(spec, "Acausal.settime")
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            st = SetTimeState{timetype(ctx)}()
            return chunkmap(
                c -> settimechunk!(st, spec, c, ctx.start, ctx.stop, false,
                    "Acausal.settime"),
                p.run(ctx),
            )
        end
    end
end
settime(p::CausalPipeline, spec) = settime(spec)(p)

end # module Acausal
