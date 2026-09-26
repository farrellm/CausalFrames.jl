# The n-ary time-interleaving source. Each input pipeline has a cursor holding
# one buffered chunk. A step picks the cursor whose head row comes first and
# claims its whole run of rows that stays ahead of the runner-up. A claim is
# just a chunk and a row range; a batch's claims are copied out together, one
# allocation per output column. Comparisons are per run, copies per batch, and
# dynamically typed work is O(inputs + ncols) per step.

"""
    merge(p::CausalPipeline, ps::CausalPipeline...; batchsize = 1024)
        -> CausalPipeline

A source running the pipelines side by side over the same context and
interleaving their rows by time. To join them end to end instead, use
[`concatenate`](@ref). There is no zero-argument form; [`emptyframe`](@ref) is
the identity.

# Arguments
- `p`, `ps...`: the pipelines. Each is evaluated over the whole context and
  clips itself. All run at once, so a merge of file sources holds every file
  open and buffers one chunk per input.

# Keywords
- `batchsize = 1024`: the minimum number of rows per emitted chunk (the last
  may be smaller). Must be positive.

The output has the union of the inputs' columns: `:time`, then the others in
the order the pipelines introduce them. A row from a pipeline lacking a column
has `missing` there. A pipeline that produces no rows contributes no columns.
Rows at equal times come in argument order, and each pipeline's own row order is
kept.

A pipeline's column names are fixed by its first chunk; a later chunk with
different names, or the same names in a different order, is an
`ArgumentError`.

```jldoctest
trades = readtable(DataFrame(time = [1, 3], price = [10.1, 10.3]))
quotes = readtable(DataFrame(time = [1, 2], bid = [10.0, 10.2]))
DataFrame(load(Context(0, 10), merge(trades, quotes)))

# output

4×3 DataFrame
 Row │ time   price      bid
     │ Int64  Float64?   Float64?
─────┼─────────────────────────────
   1 │     1       10.1  missing
   2 │     1  missing         10.0
   3 │     2  missing         10.2
   4 │     3       10.3  missing
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

# One input's read head: the buffered chunk, the next row in it, and the
# iterator behind it. `times`, the chunk's time column in the context's time
# type, is what every comparison reads, so it must stay concrete.
mutable struct MergeCursor{T}
    const index::Int                  # position in the argument list: the tie-break
    const input::PullCursor           # the input's chunks
    done::Bool                        # `input.done`, concrete: read per claim
    chunk::Union{Nothing,DataFrame}   # the buffered chunk; nothing when consumed
    times::Vector{T}                  # its time column, for ordering only
    pos::Int                          # the next unemitted row
    names::Vector{Symbol}             # this input's columns, fixed by its first chunk
    colmap::Vector{Int}               # union position -> local column, 0 = absent
    passthrough::Bool                 # names == the union names, in order
end

MergeCursor{T}(index::Int, chunks) where {T} = MergeCursor{T}(index,
    PullCursor(chunks), false, nothing, T[], 1, Symbol[], Int[], false)

# A run of rows claimed from one cursor's chunk, held by reference until the
# batch is materialized, when its rows are copied once.
struct MergePiece
    chunk::DataFrame
    colmap::Vector{Int}   # union position -> local column, 0 = absent
    lo::Int
    hi::Int
    whole::Bool           # covers the whole chunk
    passthrough::Bool     # ... and the chunk already carries the union schema
end

Base.length(pc::MergePiece) = pc.hi - pc.lo + 1

# The stateful producer behind merge's ChunkSource.
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
        # claim! pushes its piece rather than returning a Union{Nothing, _},
        # which would box it
        if !claim!(p)
            p.pendingrows == 0 && return nothing
            return materialize!(p)
        end
        p.pendingrows >= p.batchsize && return materialize!(p)
    end
end

# Run every pipeline and buffer its first chunk, fixing the union schema before
# any output (every output chunk must carry the same names). The first claim
# needs every head row anyway, so this pulls nothing extra.
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

# Advance to the next non-empty chunk, returning whether one was found.
# `convert` doesn't copy a time column already in the context's time type.
function refill!(cur::MergeCursor{T}) where {T}
    while true
        chunk = pull!(cur.input)
        if chunk === nothing
            cur.done = true
            cur.chunk = nothing
            return false
        end
        nrow(chunk) == 0 && continue
        cur.chunk = chunk
        cur.times = convert(Vector{T}, chunk.time)
        cur.pos = 1
        return true
    end
end

# O(ncols) per chunk, as in concatenate and the sinks.
function checkmergeschema!(cur::MergeCursor)
    cols = propertynames(cur.chunk::DataFrame)
    cols == cur.names || throw(ArgumentError("merge: pipeline $(cur.index) changed \
        columns mid-stream, from $(cur.names) to $(cols)"))
    return nothing
end

live(cur::MergeCursor) = cur.chunk !== nothing && cur.pos <= length(cur.times)
headtime(cur::MergeCursor) = @inbounds cur.times[cur.pos]

# A spent chunk belongs to the piece that claimed it, and then possibly to a
# consumer downstream, so the cursor drops its references.
function release!(cur::MergeCursor{T}) where {T}
    cur.chunk = nothing
    cur.times = T[]
    cur.pos = 1
    return nothing
end

# The live cursors with the smallest and next-smallest (time, index) keys, or 0
# for none. Scanning in index order with a strict `<` keeps the earlier input
# ahead at equal times. This per-claim O(inputs) loop is why the cursors sit in
# a concretely typed vector.
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
# stays below the runner-up's head key. The run is never empty, so every call
# returning true makes progress.
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
        # rows before `pos` are all <= t, so searching the whole vector lands
        # where searching pos:end would
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

# The batch's rows are copied once, into new columns of the promoted element
# type. A batch of one whole chunk adopts its column vectors instead, and goes
# downstream untouched if it already carries the union schema.
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
        # an absent column is a zero-byte Vector{Missing}; `DataFrame(frame)`
        # promotes it on concatenation
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

# The stretch of one piece's column to copy, reused across the batch. The copy
# is a dynamic call, which would box three loose `Int` arguments per piece per
# column.
mutable struct CopySpan
    off::Int
    lo::Int
    len::Int
end

copyspan!(out::AbstractVector, src::AbstractVector, span::CopySpan) =
    copyto!(out, span.off + 1, src, span.lo, span.len)

# One output column over the batch's pieces, typed as the promotion of theirs
# (Missing where a piece lacks the column). The fill runs behind a function
# barrier: one dispatch per piece, not per row.
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
