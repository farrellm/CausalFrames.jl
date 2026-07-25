# The n-ary time-interleaving source. One cursor per input pipeline, each
# holding one buffered chunk; a step picks the cursor whose head row comes
# first and claims the whole run of its rows that stays ahead of the runner-up.
# A claim moves no data — it is a chunk and a row range — and the claims of a
# batch are copied out together, one allocation per output column. So the
# comparisons are per run and the copies are per batch, never per row, and the
# only dynamically typed work (indexing the pipeline tuple, handing off a chunk
# iterator, a DataFrame column access) is O(inputs + ncols) per step.

"""
    merge(p::CausalPipeline, ps::CausalPipeline...; batchsize = 1024) -> CausalPipeline

A source running the given pipelines concurrently over the same context and
interleaving their rows by time — the time-wise merge of their outputs, as
opposed to [`concatenate`](@ref)'s end-to-end join.

The pipelines need not have the same columns. The output carries the union of
their columns, `:time` first and the rest in the order the pipelines introduce
them; a row coming from a pipeline that lacks a column carries `missing` there,
so a merged column of `T` has element type `Union{Missing, T}` once the frame
is materialized. Element *types* may differ between pipelines, as they may
between the chunks of one pipeline — `DataFrame(frame)` promotes on
concatenation.

Rows at equal times are emitted in argument order: all the tied rows of the
first pipeline, then those of the second, and so on. Each pipeline's own row
order is preserved. Every pipeline is evaluated over the whole context
`[start, stop)` and clips itself; unlike `concatenate`, they run at the same
time, so a merge of file sources holds every file open at once and buffers one
chunk per input.

A pipeline's column names are fixed by its first chunk: a later chunk whose
names differ, in content or in order, is an `ArgumentError` naming the
pipeline. A pipeline that produces no chunk at all over the window contributes
no columns, exactly as an empty right stream contributes none to
[`asofjoin`](@ref). Rows are emitted in batches of at least `batchsize` (the
trailing one excepted), so streams that alternate row by row still yield
chunk-sized output rather than a chunk per row.

There is no zero-argument form: [`emptyframe`](@ref) is the identity of
merging.

```julia
merge(readcsv("trades.csv"; types = tt), readcsv("quotes.csv"; types = qt)) |>
    filterrows(r -> ismissing(r.bid) || r.bid > 0)
```
"""
function Base.merge(p::CausalPipeline, ps::CausalPipeline...;
    batchsize::Integer = 1024)
    batchsize > 0 ||
        throw(ArgumentError("merge batchsize must be positive, got $batchsize"))
    pipelines = (p, ps...)
    n = Int(batchsize)
    return CausalPipeline(ctx::Context -> ChunkSource(MergeProducer(pipelines, ctx, n)))
end

# One input's read head: the buffered chunk, the row reached in it, and the
# lazily refilled iterator behind it. The dynamically typed fields are per-chunk
# setup state, in the shape of AsofJoinState's right stream; `times` is the
# chunk's time column as the context's time type, which is what every ordering
# comparison touches, so it is the one field that must stay concrete.
mutable struct MergeCursor{T}
    const index::Int                  # position in the argument list: the tie-break
    chunks::Any                       # the input's chunk iterator
    state::Any                        # its iteration state
    started::Bool
    done::Bool
    chunk::Union{Nothing,DataFrame}   # the buffered chunk; nothing when consumed
    times::Vector{T}                  # its time column, for ordering only
    pos::Int                          # the next unemitted row
    names::Vector{Symbol}             # this input's columns, fixed by its first chunk
    colmap::Vector{Int}               # union position -> local column, 0 = absent
    passthrough::Bool                 # names == the union names, in order
end

MergeCursor{T}(index::Int, chunks) where {T} = MergeCursor{T}(index, chunks, nothing,
    false, false, nothing, T[], 1, Symbol[], Int[], false)

# A run of rows claimed from one cursor's chunk, held until the batch is
# materialized. Nothing is copied to make a piece: the chunk it points into is
# kept alive by this reference, and its rows move exactly once, when the
# batch's columns are filled.
struct MergePiece
    chunk::DataFrame
    colmap::Vector{Int}   # union position -> local column, 0 = absent
    lo::Int
    hi::Int
    whole::Bool           # covers the whole chunk
    passthrough::Bool     # ... and the chunk already carries the union schema
end

Base.length(pc::MergePiece) = pc.hi - pc.lo + 1

# The stateful producer behind merge's ChunkSource, in the shape of
# concatenate's ConcatProducer: the pull-to-pull state lives in fields rather
# than captured locals (captured variables that are reassigned get boxed).
mutable struct MergeProducer{P<:Tuple,T}
    const pipelines::P
    const ctx::Context{T}
    const batchsize::Int
    const cursors::Vector{MergeCursor{T}}
    const pending::Vector{MergePiece}  # claimed rows, until batchsize of them
    const names::Vector{Symbol}        # the union schema, :time first
    pendingrows::Int
    initialized::Bool
end

MergeProducer(ps::P, ctx::Context{T}, batchsize::Int) where {P<:Tuple,T} =
    MergeProducer{P,T}(ps, ctx, batchsize, MergeCursor{T}[], MergePiece[], [:time],
        0, false)

function (p::MergeProducer)()
    p.initialized || initcursors!(p)
    while true
        # claim! pushes rather than returns its piece: a Union{Nothing, _} of a
        # struct holding references would be boxed, once per piece
        if !claim!(p)
            p.pendingrows == 0 && return nothing
            return materialize!(p)
        end
        p.pendingrows >= p.batchsize && return materialize!(p)
    end
end

# Run every pipeline and buffer its first chunk. The union schema has to be
# known before the first chunk goes out — a CausalFrame's chunks must all carry
# the same column names — and the merge needs every head row anyway to decide
# which one comes first, so nothing is pulled here that the first claim would
# not have pulled.
function initcursors!(p::MergeProducer{P,T}) where {P,T}
    for i in 1:length(p.pipelines)
        cur = MergeCursor{T}(i, p.pipelines[i].run(p.ctx))
        refill!(cur)
        push!(p.cursors, cur)
    end
    for cur in p.cursors
        chunk = cur.chunk
        chunk === nothing && continue
        cur.names = propertynames(chunk)
        :time in cur.names ||
            throw(ArgumentError("merge: pipeline $(cur.index) has no :time column"))
        for n in cur.names
            n in p.names || push!(p.names, n)
        end
    end
    for cur in p.cursors
        chunk = cur.chunk
        chunk === nothing && continue
        # resolved once per input: column assembly is then integer indexing
        cur.colmap = [columnindex(chunk, n) for n in p.names]
        cur.passthrough = cur.names == p.names
    end
    p.initialized = true
    return nothing
end

# Advance to the next non-empty chunk, in the shape of asofjoin's pullright!.
# Returns whether one was found. `convert` is an identity, not a copy, whenever
# the chunk's time column already has the context's time type.
function refill!(cur::MergeCursor{T}) where {T}
    while true
        cur.done && return false
        next = cur.started ? iterate(cur.chunks, cur.state) : iterate(cur.chunks)
        cur.started = true
        if next === nothing
            cur.done = true
            cur.chunk = nothing
            return false
        end
        chunk, cur.state = next
        nrow(chunk) == 0 && continue
        cur.chunk = chunk
        cur.times = convert(Vector{T}, chunk.time)
        cur.pos = 1
        return true
    end
end

# O(ncols) per chunk, the rule concatenate's checkconcat! and the sinks apply.
function checkmergeschema!(cur::MergeCursor)
    cols = propertynames(cur.chunk::DataFrame)
    cols == cur.names || throw(ArgumentError("merge: pipeline $(cur.index) changed \
        columns mid-stream, from $(cur.names) to $(cols)"))
    return nothing
end

live(cur::MergeCursor) = cur.chunk !== nothing && cur.pos <= length(cur.times)
headtime(cur::MergeCursor) = @inbounds cur.times[cur.pos]

# A spent chunk belongs to the piece that claimed it — and, once that piece is
# materialized, possibly to a consumer downstream. Drop the cursor's own
# references so nothing is read through them again.
function release!(cur::MergeCursor{T}) where {T}
    cur.chunk = nothing
    cur.times = T[]
    cur.pos = 1
    return nothing
end

# The winner is the live cursor with the smallest (time, index) key, the
# runner-up the next smallest — both by index order, so a strict `<` leaves the
# earlier argument ahead at equal times. 0 means there is none. The only
# per-claim O(inputs) loop, and the reason the cursors sit in a concretely
# typed vector: every comparison here is on T.
function pickwinner(cursors::Vector{MergeCursor{T}}) where {T}
    w = 0
    b = 0
    for i in eachindex(cursors)
        cur = cursors[i]
        live(cur) || continue
        t = headtime(cur)
        if w == 0
            w = i
        elseif t < headtime(cursors[w])
            b = w
            w = i
        elseif b == 0 || t < headtime(cursors[b])
            b = i
        end
    end
    return (w, b)
end

# Claim one piece: the longest run of the winner's rows whose (time, index) key
# stays below the runner-up's head key. The winner and the runner-up are
# distinct cursors, and equal head times imply the winner has the smaller index
# (else it would not have won), so the two branches below are the only ones.
# The run is never empty, so every call that returns true makes progress.
function claim!(p::MergeProducer)
    for cur in p.cursors
        if !cur.done && !live(cur)
            refill!(cur) && checkmergeschema!(cur)
        end
    end
    w, b = pickwinner(p.cursors)
    w == 0 && return false
    cur = p.cursors[w]
    lo = cur.pos
    hi = if b == 0
        length(cur.times)   # the last live input: the rest of its chunk
    else
        t = headtime(p.cursors[b])
        # every row before `pos` is <= t already, so the unrestricted searches
        # land in the same place as searches over pos:end would
        w < b ? searchsortedlast(cur.times, t) : searchsortedfirst(cur.times, t) - 1
    end
    cur.pos = hi + 1
    chunk = cur.chunk::DataFrame
    whole = lo == 1 && hi == length(cur.times)
    push!(p.pending,
        MergePiece(chunk, cur.colmap, lo, hi, whole, whole && cur.passthrough))
    p.pendingrows += hi - lo + 1
    # the piece holds the chunk now; the cursor need not
    whole && release!(cur)
    return true
end

# The batch's rows are copied exactly once, into freshly allocated columns of
# the promoted element type — one allocation per column, not per piece, and no
# intermediate per-piece frame to concatenate afterwards. A batch that is a
# single whole chunk skips even that: the chunk is owned and its column vectors
# are never mutated in place, so they can be adopted as they are, and a chunk
# that already carries the union schema goes downstream untouched.
function materialize!(p::MergeProducer)
    pieces = p.pending
    out =
        length(pieces) == 1 && first(pieces).whole ? adoptwhole(p, first(pieces)) :
        concatpieces(p, pieces)
    empty!(pieces)
    p.pendingrows = 0
    return out
end

function adoptwhole(p::MergeProducer, pc::MergePiece)
    pc.passthrough && return pc.chunk
    n = length(pc)
    cols = Vector{AbstractVector}(undef, length(p.names))
    for (k, j) in enumerate(pc.colmap)
        # absent columns are zero-byte Vector{Missing} (Missing is a
        # singleton); concatenation promotes them where the frame is assembled
        cols[k] = j == 0 ? Vector{Missing}(undef, n) : pc.chunk[!, j]
    end
    return DataFrame(cols, p.names; copycols = false)
end

function concatpieces(p::MergeProducer, pieces::Vector{MergePiece})
    n = p.pendingrows
    cols = Vector{AbstractVector}(undef, length(p.names))
    for k in eachindex(p.names)
        cols[k] = buildcolumn(pieces, k, n)
    end
    return DataFrame(cols, p.names; copycols = false)
end

# The stretch of one piece's column to copy. A source column's type is only
# known at run time, so the copy is a dynamic call — and a dynamic call boxes
# every non-pointer argument, which as three loose `Int`s would be three
# allocations per piece per column. Reused across the whole batch.
mutable struct CopySpan
    off::Int
    lo::Int
    len::Int
end

copyspan!(out::AbstractVector, src::AbstractVector, span::CopySpan) =
    copyto!(out, span.off + 1, src, span.lo, span.len)

# One output column over the batch's pieces: the element type is the promotion
# of the pieces' own — Missing among them wherever a piece's input lacks the
# column, which is what widens the column to Union{Missing, T}. The type is a
# runtime value here, so the filling happens behind a function barrier, where
# the output is concrete and the copies run one dispatch per piece rather than
# one per row.
function buildcolumn(pieces::Vector{MergePiece}, k::Int, n::Int)
    T = Union{}
    for pc in pieces
        j = pc.colmap[k]
        T = promote_type(T, j == 0 ? Missing : eltype(pc.chunk[!, j]))
    end
    return fillcolumn!(Vector{T}(undef, n), pieces, k)
end

function fillcolumn!(out::Vector, pieces::Vector{MergePiece}, k::Int)
    span = CopySpan(0, 0, 0)
    for pc in pieces
        j = pc.colmap[k]
        span.lo = pc.lo
        span.len = length(pc)
        if j == 0
            @inbounds for i in 1:(span.len)
                out[span.off+i] = missing
            end
        else
            copyspan!(out, pc.chunk[!, j], span)
        end
        span.off += span.len
    end
    return out
end
