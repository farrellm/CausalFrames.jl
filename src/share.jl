# Pipelines as a DAG, and the plan of one evaluation: which runs it makes, and
# how each run is shared among the operators reading it (issue #105).
#
# A pipeline is a node naming its inputs and the context it reads each one
# over, so the planner knows, before anything runs, every context each node is
# asked for. Requests a node can serve from one run are grouped into a *run
# instance* over their union, which a tee hands to each reader clipped back to
# the context it asked for. Requests with equal contexts are always grouped; a
# `startfree` node also groups requests differing in start, a `stopfree` one in
# stop, as long as their contexts overlap or touch. So a diamond (one upstream
# read by two legs of a stage) runs its upstream once instead of once per leg,
# which compounded to 2^d runs over d chained stages.
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
    const rescans::Bool    # every run reads its source from the beginning
    const sharing::Union{Nothing,Bool}  # `sharing`'s override; nothing: automatic
end

# The builders run once per pipeline built and never per chunk, so none of
# them specializes: every pipeline's run function and inputs are distinct
# types, and each would otherwise compile its own copy of this plumbing.

# A source: `run(ctx)` builds its chunks. `rescans` marks one whose every run
# reads from the beginning of its file, which the plan never splits.
pipesource(@nospecialize(run); startfree::Bool = true, stopfree::Bool = true,
    rescans::Bool = false) =
    CausalPipeline(SourceRun(run), CausalPipeline[], identity, startfree,
        stopfree, rescans, nothing)

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
    return CausalPipeline(run, ins, contexts, startfree, stopfree, false, sharing)
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
depend on where the context starts (or stops) and only that differs, as long as
the contexts overlap or touch. Readers whose windows leave a gap get a run
each, unless that would read rows twice: when the pipeline's look-back bridges
the gap, or when its source reads from its beginning on every run (a
[`readcsv`](@ref) or [`readjls`](@ref) file, some [`readparquet`](@ref) and
[`readtable`](@ref) reads, or a [`CausalPipeline`](@ref) declaring `rescans`).
Files and tables don't depend on the context, except for the rows at `stop`
that `closed = true` keeps, and most transforms keep their input's
independence. A [`clock`](@ref), [`head`](@ref), a cumulative
[`addsummarycolumns`](@ref), or a
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
    iter::Any          # what its reader was handed; nothing until opened
    abandoned::Bool    # its reader won't pull again
    RunSlot(ctx::Context, isolated = nothing) =
        new(ctx, isolated, nothing, 0, nothing, false)
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

# What an operator's `run` reaches its inputs through (`openinput`), over a
# planned run (`RunInputs`) or a direct one.
abstract type PipelineInputs end

struct RunInputs <: PipelineInputs
    instance::RunInstance
end

# Input i's chunks, over the context the operator declared for it.
openinput(ins::RunInputs, i::Int) = openslot!(ins.instance.inputs[i])
ninputs(ins::RunInputs) = length(ins.instance.inputs)

# Input i's context, from what a node's `contexts` gave.
inputcontext(cs, i::Int) = cs isa Context ? cs : cs[i]

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

# What an operator's `run` reaches its inputs through, run directly: the node,
# and its inputs' contexts.
struct DirectInputs <: PipelineInputs
    node::CausalPipeline
    contexts::Any
end

openinput(ins::DirectInputs, i::Int) =
    opendirect(ins.node.inputs[i], inputcontext(ins.contexts, i))
ninputs(ins::DirectInputs) = length(ins.node.inputs)

opendirect(@nospecialize(p::CausalPipeline), ctx::Context) =
    p.run(ctx, DirectInputs(p, p.contexts(ctx)))

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
            cs = node.contexts(inst.ctx)
            iso = node.sharing === false ? inst : first(inst.slots).isolated
            for (i, input) in pairs(node.inputs)
                slot = RunSlot(inputcontext(cs, i), iso)
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

# Group a node's requests into runs by `sharekey`, the groups in request order,
# then split each group where its contexts leave a gap (`pushruns!`). Most
# nodes of a plan have one request, which is one run whatever its key.
function instances(@nospecialize(node::CausalPipeline), slots::Vector{RunSlot})
    length(slots) == 1 && return [newinstance(node, slots)]
    groups = Pair{Any,Vector{RunSlot}}[]
    for s in slots
        k = sharekey(node, s)
        g = findfirst(gr -> isequal(first(gr), k), groups)
        g === nothing ? push!(groups, k => [s]) : push!(last(groups[g]), s)
    end
    insts = RunInstance[]
    for gr in groups
        pushruns!(insts, node, last(gr))
    end
    return insts
end

# An instance per run of overlapping or touching contexts in the group, by
# start, so no run covers a gap none of its slots asked for (issue #132). A
# forced share and a `rescans` source are always one run. Runs whose input
# contexts overlap or touch again (a look-back bridging the gap) stay one too:
# apart, the node would read those input rows once per run. A single slot, or
# a group fixing its start or its stop, can't leave a gap, so it skips the
# sort. Unsplit, the group (sorted in place) is the instance's own slot
# vector, so the common case allocates nothing more.
function pushruns!(insts::Vector{RunInstance}, @nospecialize(node::CausalPipeline),
    slots::Vector{RunSlot})
    (
        node.sharing === true || node.rescans || !(node.startfree & node.stopfree) ||
        length(slots) == 1
    ) && return push!(insts, newinstance(node, slots))
    ctx = first(slots).ctx
    # dispatched on the erased context, so a group too wide for insertion
    # compiles its merge only when one is planned
    length(slots) > SORTINSERTMAX ? mergeslots!(slots, ctx) : sortslots!(slots, ctx)
    n, lo, hi = length(slots), 1, runend(slots, 1, ctx)
    hi == n && return push!(insts, newinstance(node, slots))
    # a later cluster starts past every earlier stop, so the hull of the
    # clusters `lo:hi` ends where the last of them does
    hull = slothull(slots, lo, hi, ctx)
    while hi < n
        next = runend(slots, hi + 1, ctx)
        nexthull = slothull(slots, hi + 1, next, ctx)
        if inputsmeet(node, hull, nexthull)
            hull = Context(hull.start, nexthull.stop)
        else
            push!(insts, newinstance(node, slots[lo:hi]))
            lo, hull = hi + 1, nexthull
        end
        hi = next
    end
    return push!(insts, newinstance(node, lo == 1 ? slots : slots[lo:n]))
end

# The union of the sorted slots `lo:hi`'s contexts.
function slothull(slots::Vector{RunSlot}, lo::Int, hi::Int, ::Context{T}) where {T}
    stop = (slots[lo].ctx::Context{T}).stop
    for i in (lo+1):hi
        stop = max(stop, (slots[i].ctx::Context{T}).stop)
    end
    return Context((slots[lo].ctx::Context{T}).start, stop)
end

# Whether runs over `a` and the later `b` would read any input over
# overlapping or touching contexts. Called once per gap, so it may dispatch.
function inputsmeet(@nospecialize(node::CausalPipeline), a::Context, b::Context)
    as, bs = node.contexts(a), node.contexts(b)
    for i in eachindex(node.inputs)
        ia, ib = inputcontext(as, i), inputcontext(bs, i)
        ia.start <= ib.stop && ib.start <= ia.stop && return true
    end
    return false
end

# Sort a group by start, by a stable insertion sort with typed comparisons: a
# group is usually a handful of slots of one time type (it is in the key), and
# `sort!` over the erased contexts dispatched per comparison. It takes no node,
# whose `@nospecialize` slowed that dispatch tenfold.
function sortslots!(slots::Vector{RunSlot}, ::Context{T}) where {T}
    for i in 2:length(slots)
        s = slots[i]
        t = (s.ctx::Context{T}).start
        j = i - 1
        while j >= 1 && (slots[j].ctx::Context{T}).start > t
            slots[j+1] = slots[j]
            j -= 1
        end
        slots[j+1] = s
    end
    return slots
end

# The widest group `sortslots!` sorts: insertion is quadratic, and past about
# this many slots the merge is faster.
const SORTINSERTMAX = 32

# Sort a wide fan-out's group like `sortslots!`, by a stable bottom-up merge
# sort through one buffer. `sortperm` of the typed starts sorted faster but
# compiled three times longer.
function mergeslots!(slots::Vector{RunSlot}, ::Context{T}) where {T}
    n, w = length(slots), 1
    src, dst = slots, similar(slots)
    while w < n
        for lo in 1:2w:n
            mid, hi = min(lo + w - 1, n), min(lo + 2w - 1, n)
            i, j = lo, mid + 1
            for k in lo:hi
                if j > hi || (
                    i <= mid &&
                    (src[i].ctx::Context{T}).start <= (src[j].ctx::Context{T}).start
                )
                    dst[k] = src[i]
                    i += 1
                else
                    dst[k] = src[j]
                    j += 1
                end
            end
        end
        src, dst, w = dst, src, 2w
    end
    return src === slots ? slots : copyto!(slots, src)
end

# The last slot of the run of overlapping or touching contexts from `lo`.
function runend(slots::Vector{RunSlot}, lo::Int, ::Context{T}) where {T}
    stop = (slots[lo].ctx::Context{T}).stop
    for i in (lo+1):length(slots)
        c = slots[i].ctx::Context{T}
        c.start > stop && return i - 1
        stop = max(stop, c.stop)
    end
    return length(slots)
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
    s.iter === nothing || throw(ArgumentError("internal: a run slot was opened twice"))
    inst = s.instance::RunInstance
    if length(inst.slots) == 1
        s.iter = runinstance(inst)
    else
        inst.tee === nothing && (inst.tee = ChunkTee(inst))
        s.iter = ChunkSource(TeeSide(inst.tee::ChunkTee, s.side))
    end
    return s.iter
end

runinstance(inst::RunInstance) =
    InstanceRun(inst, inst.node.run(inst.ctx, RunInputs(inst)), false)

# The slot's reader won't pull again: abandon what it was handed, or, never
# opened, keep its run's other readers from holding chunks for it. A run whose
# every slot is abandoned unopened never starts, so its own input slots are
# abandoned too: otherwise a shared run below it would queue its whole output
# for a reader that will never come (`concatenate`'s unreached inputs).
function abandonslot!(s::RunSlot)
    s.iter === nothing || return abandon!(s.iter)
    s.abandoned && return nothing
    inst = s.instance::RunInstance
    if inst.tee === nothing
        s.abandoned = true
        all(t -> t.abandoned, inst.slots) && foreach(abandonslot!, inst.inputs)
    else
        abandonside!(inst.tee::ChunkTee, s.side)
    end
    return nothing
end

# A planned run's chunks. Once they end, or their reader abandons them, the
# operator won't open or pull its inputs again, so every input slot is
# abandoned: a run a slot shares stops holding chunks back for it, opened or
# not. So no operator wires up abandoning of its own; only one stopping early
# while it still runs (`head`) abandons its input itself.
mutable struct InstanceRun
    const instance::RunInstance
    const iter::Any    # erased, so a run's type never nests its inputs'
    done::Bool
end

Base.IteratorSize(::Type{InstanceRun}) = Base.SizeUnknown()
Base.eltype(::Type{InstanceRun}) = DataFrame

function Base.iterate(r::InstanceRun, state...)
    next = iterate(r.iter, state...)
    next === nothing && (abandon!(r); return nothing)
    c, s = next
    return (c::DataFrame, s)
end

# A side's view of the shared run: the rows of its own context. The run's
# context is the union, so a side starts later (the lead-in dropped as
# `warmup` drops it) or stops earlier (`cutstop`), or neither.
mutable struct SideClip{T}
    const leadin::LeadInDrop{T}  # passed from the outset unless it starts later
    const stop::T
    const cutstop::Bool
    finished::Bool     # a row at or after `stop` has been seen: nothing more
end
SideClip(side::Context{T}, run::Context{T}) where {T} =
    SideClip{T}(LeadInDrop{T}(side.start, side.start <= run.start), side.stop,
        side.stop < run.stop, false)

# The side's part of the chunk, or `nothing`.
function clipside!(cl::SideClip, c::DataFrame)
    cl.finished && return nothing
    hi = nrow(c)
    if cl.cutstop
        hi = firstkept(c.time, cl.stop) - 1
        hi < nrow(c) && (cl.finished = true)
    end
    return dropleadin!(cl.leadin, c, hi)
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
# holding chunks back for it. What a planned run hands out (an `InstanceRun`, a
# tee side) acts on it, and a `PullCursor` passes it on; anything else is only
# ever run directly, where nothing is shared, so it is a no-op.
abandon!(::Any) = nothing
function abandon!(c::PullCursor)
    c.done && return nothing
    c.done = true
    return abandon!(c.iter)
end
function abandon!(r::InstanceRun)
    r.done && return nothing
    r.done = true
    foreach(abandonslot!, r.instance.inputs)
    return nothing
end
abandon!(it::ChunkSource{<:TeeSide}) = abandonside!(it.produce.tee, it.produce.side)
