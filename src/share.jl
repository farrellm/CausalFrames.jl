# Pipelines as a DAG, and the plan of one evaluation: which runs it makes, and
# how each run is shared among the operators reading it (issue #105).
#
# A pipeline is a node naming its inputs and the context it reads each one
# over, so the planner knows, before anything runs, every context each node is
# asked for. Requests a node can serve from one run are grouped into a *run
# instance* over their union, which a tee hands to each reader clipped back to
# the context it asked for. Requests with equal contexts are always grouped; a
# `startfree` node also groups requests differing in start, a `stopfree` one in
# stop. So a diamond (one upstream read by two legs of a stage) runs its
# upstream once instead of once per leg, which compounded to 2^d runs over d
# chained stages.
#
# The run instance is also the unit a parallel executor would schedule:
# operators reach their inputs only through `openinput`, and all of a run's
# state lives in its instances and tees, never in the nodes or in globals.
# Pulling order (and so any prefetching) changes memory and timing, never the
# rows.

# Built by `pipesource`, `pipenode` and `pipetransform`; the public constructor
# (a source) and its docstring are in pipeline.jl. Immutable after
# construction: identity is the node's, cheap to hash, and planning never
# writes to it.
mutable struct CausalPipeline{F}
    const run::F       # (ctx, ins::RunInputs) -> iterator of DataFrame chunks
    const inputs::Vector{CausalPipeline}  # erased, so types don't nest
    const contexts::Any  # ctx -> each input's context (a Context: all of them)
    const startfree::Bool  # rows don't depend on where the run starts
    const stopfree::Bool   # rows don't depend on where the run stops
    const sharing::Union{Nothing,Bool}  # `sharing`'s override; nothing: automatic
end

# The builders run once per pipeline built and never per chunk, so none of
# them specializes: every pipeline's run function and inputs are distinct
# types, and each would otherwise compile its own copy of this plumbing.

# A source: `run(ctx)` builds its chunks.
pipesource(@nospecialize(run); startfree::Bool = true, stopfree::Bool = true) =
    CausalPipeline(SourceRun(run), CausalPipeline[], nothing, startfree,
        stopfree, nothing)

# A source's run function, ignoring the (absent) inputs.
struct SourceRun{R}
    run::R
end
(s::SourceRun)(ctx::Context, _) = s.run(ctx)

# An operator over `inputs`: `run(ctx, ins)` builds its chunks, opening input i
# with `openinput(ins, i)` (each at most once); `contexts(ctx)` gives the
# context each input is read over, a single `Context` for all of them. Its own
# `startfree`/`stopfree` say whether the operator keeps its inputs' freedom;
# the node's are those and-ed with every input's.
function pipenode(@nospecialize(run), @nospecialize(inputs); contexts = identity,
    startfree::Bool = true, stopfree::Bool = true,
    sharing::Union{Nothing,Bool} = nothing)
    ins = CausalPipeline[]
    for p in inputs
        p = p::CausalPipeline
        push!(ins, p)
        startfree &= p.startfree
        stopfree &= p.stopfree
    end
    return CausalPipeline(run, ins, contexts, startfree, stopfree, sharing)
end

# A unary transform's shell: `run(upstream, ctx)` builds its chunks from the
# input's, read over `contexts(ctx)`. Written as a do block, so a transform
# reads `pipetransform() do upstream, ctx … end`.
pipetransform(@nospecialize(run); contexts = identity, startfree::Bool = true,
    stopfree::Bool = true) =
    UnaryTransform(UnaryRun(run), contexts, startfree, stopfree)

# The transform `pipetransform` returns, and its node's run function.
struct UnaryTransform
    run::Any
    contexts::Any
    startfree::Bool
    stopfree::Bool
end
(t::UnaryTransform)(@nospecialize(p::CausalPipeline)) =
    pipenode(t.run, (p,); t.contexts, t.startfree, t.stopfree)

struct UnaryRun{R}
    run::R
end
(u::UnaryRun)(ctx::Context, ins) = u.run(openinput(ins, 1), ctx)

"""
    sharing(force::Bool) -> (CausalPipeline -> CausalPipeline)
    sharing(p::CausalPipeline, force::Bool) -> CausalPipeline

A transform passing its input through unchanged, but deciding whether the
operators reading the result share one run of it.

By default a pipeline read in several places of one evaluation, such as both
sides of `asofjoin(p, p |> f)` or the input of [`addrollingcolumns`](@ref),
runs once for all of them when that gives each reader the rows it would get
alone: when they ask for the same context, or when the pipeline's rows don't
depend on where the context starts (or stops) and only that differs. Files and
tables don't, and most transforms keep their input's independence. A
[`clock`](@ref), [`head`](@ref), a cumulative [`addsummarycolumns`](@ref), or a
[`summarize`](@ref) emitting at `stop` do depend on it, so a reader asking for
a widened context gets a run of its own.

# Arguments
- `force`: `true` shares one run among every reader, over the union of their
  contexts, each reader seeing that run clipped to its own context: rows that
  may differ from its own run's when the pipeline depends on the context.
  `false` gives every reader its own run of the pipeline and of everything
  upstream of it, even over the same context, which holds no chunks back for
  a reader that lags far behind the others.

```jldoctest
p = clock(1) |> addsummarycolumns(Count())
t = addrollingcolumns((w2 = 2,), Sum(:count))
DataFrame(load(Context(3, 6), p |> sharing(true) |> t))

# output

3×3 DataFrame
 Row │ time   count  w2_count_sum
     │ Int64  Int64  Int64
─────┼────────────────────────────
   1 │     3      3             6
   2 │     4      4             9
   3 │     5      5            12
```

Here the counts depend on where the run starts. Sharing the run widened by the
look-back shows the counts from 1, the ones summed; without `sharing(true)` the
rows come from their own run over `[3, 6)` and restart at 1.
"""
sharing(force::Bool) =
    (@nospecialize(p::CausalPipeline),) -> pipenode(PassRun(), (p,); sharing = force)

struct PassRun end
(::PassRun)(ctx::Context, ins) = openinput(ins, 1)
sharing(@nospecialize(p::CausalPipeline), force::Bool) = sharing(force)(p)

# --- planning --------------------------------------------------------------

# One reader's request of a node: the context it asked for, and once its node
# is planned, the instance serving it and its side in that instance's tee.
# `isolated` is the `sharing(false)` run the request is made under, if any:
# requests under different ones never share, so such a run's whole upstream is
# its own.
mutable struct RunSlot
    const ctx::Context
    const isolated::Any  # nothing, or the isolating RunInstance
    instance::Any      # its RunInstance
    side::Int
    opened::Bool
    abandoned::Bool    # its reader won't pull again
    RunSlot(ctx::Context, isolated = nothing) =
        new(ctx, isolated, nothing, 0, false, false)
end

# One run of a node, over the union of its slots' contexts, and the slots it
# reads its own inputs through.
mutable struct RunInstance
    const node::CausalPipeline
    const ctx::Context
    const slots::Vector{RunSlot}
    const inputs::Vector{RunSlot}
    tee::Any           # its ChunkTee once a slot opens; one slot needs none
end

# What an operator's `run` reaches its inputs through: `openinput` and
# `abandoninput`, over a planned run (`RunInputs`) or a direct one.
abstract type PipelineInputs end

struct RunInputs <: PipelineInputs
    instance::RunInstance
end

# Input i's chunks, over the context the operator declared for it.
openinput(ins::RunInputs, i::Int) = openslot!(ins.instance.inputs[i])

# For an operator that will never open input i, which would otherwise have
# its run's other readers hold chunks for it. One that opened it abandons the
# iterator instead (`abandon!`).
abandoninput(ins::RunInputs, i::Int) = abandonslot!(ins.instance.inputs[i])

# The chunks of `p` over `ctx`: plan the evaluation and open its root.
runchunks(ctx::Context, @nospecialize(p::CausalPipeline)) =
    istree(p) ? opendirect(p, ctx) : openslot!(planrun(p, ctx))

# Without a node reached twice there is nothing to share: each node is asked
# for one context, by one reader, so it runs directly, saving the plan's
# allocations (a few microseconds per evaluation, all of a tiny one's cost).
# The visited nodes are kept in a vector: pipelines are small, and a
# `Set` costs more than the scans it saves.
istree(@nospecialize(p::CausalPipeline)) = istree!(CausalPipeline[], p)
function istree!(seen::Vector{CausalPipeline}, @nospecialize(p::CausalPipeline))
    any(q -> q === p, seen) && return false
    push!(seen, p)
    for q in p.inputs
        istree!(seen, q) || return false
    end
    return true
end

# What an operator's `run` reaches its inputs through, run directly.
struct DirectInputs <: PipelineInputs
    node::CausalPipeline
    ctx::Context
end

function openinput(ins::DirectInputs, i::Int)
    cs = ins.node.contexts(ins.ctx)
    return opendirect(ins.node.inputs[i], cs isa Context ? cs : cs[i])
end
abandoninput(::DirectInputs, ::Int) = nothing   # never opened, never run

opendirect(@nospecialize(p::CausalPipeline), ctx::Context) =
    p.run(ctx, DirectInputs(p, ctx))

# Visits the nodes after everything reading them (reverse postorder), so a
# node's requests are all known when it is planned, and passes each instance's
# input contexts down as requests of its inputs.
function planrun(@nospecialize(root::CausalPipeline), ctx::Context)
    order = CausalPipeline[]
    postorder!(order, IdDict{CausalPipeline,Nothing}(), root)
    requests = IdDict{CausalPipeline,Vector{RunSlot}}()
    rootslot = RunSlot(ctx)
    requests[root] = [rootslot]
    for node in Iterators.reverse(order)
        for inst in instances(node, requests[node])
            cs = node.contexts === nothing ? nothing : node.contexts(inst.ctx)
            iso = node.sharing === false ? inst : first(inst.slots).isolated
            for (i, input) in pairs(node.inputs)
                slot = RunSlot(cs isa Context ? cs : cs[i], iso)
                inst.inputs[i] = slot
                push!(get!(() -> RunSlot[], requests, input), slot)
            end
        end
    end
    return rootslot
end

function postorder!(order::Vector{CausalPipeline}, seen::IdDict,
    @nospecialize(p::CausalPipeline))
    haskey(seen, p) && return order
    seen[p] = nothing
    for q in p.inputs
        postorder!(order, seen, q)
    end
    return push!(order, p)
end

# Group a node's requests into runs by `sharekey`, in request order.
function instances(@nospecialize(node::CausalPipeline), slots::Vector{RunSlot})
    groups = Pair{Any,Vector{RunSlot}}[]
    for s in slots
        k = sharekey(node, s)
        g = findfirst(gr -> isequal(first(gr), k), groups)
        g === nothing ? push!(groups, k => [s]) : push!(last(groups[g]), s)
    end
    return [newinstance(node, last(gr)) for gr in groups]
end

# Requests share a run when their keys are equal. The time type is always in
# the key, since a source converts its rows to it, and so is the isolating
# run. A slot is a key of its own.
function sharekey(@nospecialize(node::CausalPipeline), s::RunSlot)
    node.sharing === false && return s
    node.sharing === true && return (s.isolated, typeof(s.ctx))
    return (s.isolated, typeof(s.ctx), node.startfree ? nothing : s.ctx.start,
        node.stopfree ? nothing : s.ctx.stop)
end

function newinstance(@nospecialize(node::CausalPipeline), slots::Vector{RunSlot})
    ctx = Context(minimum(s -> s.ctx.start, slots), maximum(s -> s.ctx.stop, slots))
    inst = RunInstance(node, ctx, slots,
        Vector{RunSlot}(undef, length(node.inputs)), nothing)
    for (j, s) in pairs(slots)
        s.instance = inst
        s.side = j
    end
    return inst
end

# --- running ---------------------------------------------------------------

# A lone slot reads its instance's run directly; several read it through a tee
# built when the first of them opens.
function openslot!(s::RunSlot)
    s.opened && throw(ArgumentError("internal: a run slot was opened twice"))
    s.opened = true
    inst = s.instance::RunInstance
    length(inst.slots) == 1 && return runinstance(inst)
    inst.tee === nothing && (inst.tee = ChunkTee(inst))
    return ChunkSource(TeeSide(inst.tee::ChunkTee, s.side))
end

runinstance(inst::RunInstance) = inst.node.run(inst.ctx, RunInputs(inst))

function abandonslot!(s::RunSlot)
    s.abandoned = true
    inst = s.instance::RunInstance
    inst.tee === nothing || abandonside!(inst.tee::ChunkTee, s.side)
    return nothing
end

# A side's view of the shared run: the rows of its own context. The run's
# context is the union, so a side starts later (`cutstart`, the lead-in
# dropped as `warmup` drops it) or stops earlier (`cutstop`), or neither.
mutable struct SideClip{T}
    const start::T
    const stop::T
    const cutstart::Bool
    const cutstop::Bool
    passed::Bool       # a row at or after `start` has been kept
    finished::Bool     # a row at or after `stop` has been seen: nothing more
end
SideClip(side::Context{T}, run::Context{T}) where {T} =
    SideClip{T}(side.start, side.stop, side.start > run.start,
        side.stop < run.stop, false, false)

# The side's part of the chunk, or `nothing`. A chunk kept whole is passed on
# as is, a straddling one sliced, never `deleteat!`ed (the `writecsv`
# hand-off argument).
function clipside!(cl::SideClip, c::DataFrame)
    cl.finished && return nothing
    lo = 1
    if cl.cutstart && !cl.passed
        lo = firstkept(c.time, cl.start)
        lo > nrow(c) && return nothing
        cl.passed = true
    end
    hi = nrow(c)
    if cl.cutstop
        hi = firstkept(c.time, cl.stop) - 1
        hi < nrow(c) && (cl.finished = true)
    end
    lo > hi && return nothing
    return lo == 1 && hi == nrow(c) ? c : c[lo:hi, :]
end

# One run read by several sides at their own paces. A side with queued chunks
# reads its queue; otherwise it pulls the run and queues its part of the chunk
# for every other live side, as a private index over the same vectors (sound by
# the `writecsv` hand-off argument), so each side owns what it is given. A
# queue holds only what its side is behind by, and only the rows it keeps: a
# side clipped to a later start queues none of the lead-in. A side that is
# abandoned, or past its stop, queues nothing more, and once no side is live
# the run itself is abandoned.
struct ChunkTee{T}
    upstream::PullCursor
    slots::Vector{RunSlot}
    clips::Vector{SideClip{T}}
    queues::Vector{Vector{DataFrame}}
end
function ChunkTee(inst::RunInstance)
    clips = [SideClip(s.ctx, inst.ctx) for s in inst.slots]
    queues = [DataFrame[] for _ in inst.slots]
    return ChunkTee(PullCursor(runinstance(inst)), inst.slots, clips, queues)
end

struct TeeSide{T}
    tee::ChunkTee{T}
    side::Int
end
(s::TeeSide)() = teepull!(s.tee, s.side)

islive(tee::ChunkTee, j::Int) = !tee.slots[j].abandoned && !tee.clips[j].finished

function teepull!(tee::ChunkTee, side::Int)
    q = tee.queues[side]
    isempty(q) || return popfirst!(q)
    while islive(tee, side)
        c = pull!(tee.upstream)
        c === nothing && return nothing
        c = c::DataFrame
        for j in eachindex(tee.slots)
            (j == side || !islive(tee, j)) && continue
            k = clipside!(tee.clips[j], c)
            k === nothing || push!(tee.queues[j], DataFrame(k; copycols = false))
        end
        k = clipside!(tee.clips[side], c)
        checklive!(tee)
        k === nothing || return k
    end
    return nothing
end

function abandonside!(tee::ChunkTee, side::Int)
    tee.slots[side].abandoned = true
    empty!(tee.queues[side])
    checklive!(tee)
    return nothing
end

checklive!(tee::ChunkTee) =
    any(j -> islive(tee, j), eachindex(tee.slots)) || abandon!(tee.upstream)

# --- abandoning ------------------------------------------------------------

# Tell an input iterator its reader won't pull it again, so a shared run stops
# holding chunks back for it: `head` once its budget is spent, a join once its
# left input ends. Forwarded through the iterators and producers that hold an
# input; a no-op for anything else.
abandon!(::Any) = nothing
function abandon!(c::PullCursor)
    c.done && return nothing
    c.done = true
    return abandon!(c.iter)
end
abandon!(it::ChunkSource) = abandon!(it.produce)
abandon!(s::TeeSide) = abandonside!(s.tee, s.side)
