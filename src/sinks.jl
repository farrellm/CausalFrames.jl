# The background writer every file sink (writecsv, writeparquet, writejls)
# hands its chunks to, and the pass-through transform wiring it in.

checkqueue(queue::Integer, op::String) = checknonnegative(queue, "$op queue")

# Per-run writer state for every file sink: the queue feeding the background
# task, and the first chunk's column names, which fix the file's columns.
# `label` names the operator in errors.
mutable struct ChunkSink
    const chan::Channel{DataFrame}
    const task::Task
    const label::String
    names::Union{Nothing,Vector{String}}
end

# Every file sink's transform: a pass-through chunkmap handing each chunk to a
# `ChunkSink` built by `makesink()` when the run starts (truncating the file
# then), and joining its writer once upstream is exhausted. `SinkGuard` joins it
# too when the run fails instead.
function sinktransform(makesink)
    return pipetransform() do p, ctx
        sink = makesink()
        return SinkGuard(
            chunkmap(c -> sinkchunk(sink, c), p.run(ctx);
                flush = () -> finishwrite!(sink)),
            sink)
    end
end

# A file sink's chunks, releasing the writer when a run fails: an error from
# upstream, from the sink's own checks or from the writer itself would otherwise
# leave the writer blocked on its queue, holding the file open, for good.
struct SinkGuard{I}
    inner::I
    sink::ChunkSink
end

Base.IteratorSize(::Type{<:SinkGuard}) = Base.SizeUnknown()
Base.eltype(::Type{<:SinkGuard}) = DataFrame

function Base.iterate(g::SinkGuard, state...)
    try
        return iterate(g.inner, state...)
    catch e
        abortwrite!(g.sink, e)
        rethrow()
    end
end

function ChunkSink(writeloop, queue::Int, label::String)
    chan = Channel{DataFrame}(queue)
    task = Threads.@spawn writeloop(chan)
    # A failed writer closes the channel with its exception, so the pipeline
    # task sees it at the next put! rather than deadlocking on a full queue.
    bind(chan, task)
    return ChunkSink(chan, task, label, nothing)
end

function sinkchunk(sink::ChunkSink, c::DataFrame)
    sink.names = samecolumns(sink.names, c, sink.label)
    put!(sink.chan, c)
    # The writer reads `c` concurrently while downstream may mutate its chunk's
    # column index, so hand downstream a private index over the same vectors,
    # which are never mutated in place. O(ncols).
    return DataFrame(c; copycols = false)
end

# Called once, when upstream is exhausted: close the queue and join the writer,
# so the file is complete and closed by the time the stream ends.
function finishwrite!(sink::ChunkSink)
    close(sink.chan)
    wait(sink.task)
    return nothing
end

# Called when the run fails: close the queue with the run's exception, so the
# writer stops with it once it has drained the queued chunks (closing its file,
# but writing no parquet footer), and join it. The writer's own failure is
# already the run's, or subsumed by it, so it is not reported again.
function abortwrite!(sink::ChunkSink, e)
    close(sink.chan, e isa Exception ? e : ErrorException(string(e)))
    try
        wait(sink.task)
    catch
    end
    return nothing
end
