# Acausal operations. Everything here looks *forward* in time, violating the
# causal invariant the rest of the package upholds, so it lives in its own
# submodule and is reached only through `using CausalFrames.Acausal` — never
# re-exported from the top level. The forward join runs on `asofjoin`'s join
# engine (`JoinConfig`, `JoinState`, `joinchunk!` in join.jl) and supplies only
# its direction: a store and a kernel picked by dispatch on `Forward`, with the
# match direction, the tie-break, and the context widening all inverted.
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

# The mirror of `widenstart`: forward tolerance widens the right
# window forward, so lookahead past the window end is covered — the one place
# times are added. The explicit guard rejects a negative tolerance, which the
# Context constructor (only start <= stop) would otherwise accept.
futurecontext(ctx::Context, ::Nothing) = ctx
function futurecontext(ctx::Context, tolerance)
    stop = ctx.stop + tolerance
    stop >= ctx.stop || throw(ArgumentError(
        "futurejoin tolerance must be non-negative, got $tolerance"))
    return Context(ctx.start, stop)
end

# The forward direction of `asofjoin`'s join engine (`JoinConfig`, `JoinState`,
# `joinchunk!`): the same driver, with this store and kernel picked by dispatch
# on `Forward`, and `cmp` being `>` or `>=`.
struct Forward end

# A per-key FIFO of buffered future right rows: append on pull, logical
# pop-front by advancing `head` (the segtree.jl idiom), with an amortized
# compaction so a buffer that stays small cannot grow its backing vector
# without bound. `matches` stores copied row values, not buffer indices, so
# compaction never invalidates an emitted match.
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

# Keyed by K over rows V, as the backward `SlotStore` is; a mutable
# `KeyBuffer` already answers a lookup with a pointer, so there is no
# `Union{Nothing,V}` to box. The matches buffer and the per-key buffers may be
# half-consumed mid-left-chunk when a widening arrives, so both convert.
newstore(::Forward, ::Type{K}, ::Type{V}) where {K,V} = Dict{K,KeyBuffer{V}}()
widenstore(store::Dict{<:Any,<:KeyBuffer}, ::Type{K}, ::Type{V}) where {K,V} =
    Dict{K,KeyBuffer{V}}(
        convert(K, k) => KeyBuffer{V}(convert(Vector{V}, b.rows), b.head)
        for (k, b) in store)

# --- merge kernel ----------------------------------------------------------
#
# Called with concretely typed arguments; the per-row work compiles down to
# direct column access. On entry it drains the current right chunk fully into
# the per-key buffers (a forward join needs rows *ahead* of t, so admission is
# not gated by t), then processes left rows from index i. Returns
# (i, rpos, needpull): needpull means a left row's key has no buffered future
# row yet but the right stream is not exhausted — the driver must pull the
# next right chunk before row i can be resolved.
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
        # Drop buffered rows now before t (strict: at or before t); they can
        # never match this or any later (larger-t) left row. `after` is `>`
        # or `>=`, so its negation is the discard test — the reverse of
        # asofjoin's `before`, never reused.
        if b !== nothing
            while !bufempty(b) && !after(buffront(b).time, t)
                bufpop!(b)
            end
        end
        if b === nothing || bufempty(b)
            # No buffered future row for this key. If the right stream may
            # still hold one, pull and re-enter at i; otherwise it is missing.
            rdone || return (i, rpos, true)
        else
            # The front is the earliest right row with time >= t (strict >).
            # Tolerance staleness is decided here, against each left row: a
            # front too far ahead now may match a later, larger t, so never
            # evict on tolerance — leave the slot missing.
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

# The mirror of lag's lagcontext: the whole window slides forward by the offset
# so the -offset shift lands the output back in [start, stop). The guard rejects
# a negative offset (which the Context constructor, only start <= stop, would
# otherwise accept and silently turn into a causal lag).
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
