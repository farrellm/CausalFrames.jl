# The lookup join: a key-only join against an in-memory table with no time
# column. A table without time is constant over the window, so there is nothing
# to stream or merge — the table is resolved once, at construction, into a
# `Dict{K,Int}` row index over copied columns, and the transform is a stateless
# chunkmap. Per chunk the type-unstable work (the column table, the schema
# checks) happens once; the per-row lookup and the per-column gathers are
# function barriers over concretely typed arguments.

"""
    lookupjoin(table; key, unmatched = :missing, leftprefix = nothing,
               rightprefix = nothing) -> (CausalPipeline -> CausalPipeline)
    lookupjoin(p::CausalPipeline, table; key, ...) -> CausalPipeline

A transform joining each row to the row of `table` with the same key, appending
`table`'s other columns. Having no time, `table` holds for the whole window.
To join data that changes over time, use [`asofjoin`](@ref).

# Arguments
- `table`: any Tables.jl table without a `:time` column (a `DataFrame`, a
  `CSV.File`, …), holding at most one row per key; a duplicate key is an
  `ArgumentError`. It is validated and copied when `lookupjoin` is called. An
  empty table still appends its columns.

# Keywords
- `key`: required; a column name or collection of column names, present in both
  `table` and the input. Keys match by `isequal`, so `1` matches `1.0` and
  `missing` matches `missing`. Key columns appear once, from the input, never
  prefixed.
- `unmatched = :missing`: what to do with an input row whose key is not in
  `table` — `:missing` fills the appended columns with `missing` (so they become
  `Union{Missing, T}`), `:error` throws an `ArgumentError`, and `:drop` drops the
  row. Under `:error` and `:drop` the appended columns keep `table`'s types.
- `leftprefix = nothing`, `rightprefix = nothing`: rename that side's non-time,
  non-key columns to `"{prefix}_{name}"`. Output names must be unique.

```jldoctest
trades = readtable(DataFrame(time = [1, 2], sym = ["a", "b"], qty = [100, 200]))
dim = DataFrame(sym = ["a", "b"], sector = ["tech", "energy"])
DataFrame(load(Context(0, 10), trades |> lookupjoin(dim; key = :sym)))

# output

2×4 DataFrame
 Row │ time   sym     qty    sector
     │ Int64  String  Int64  String?
─────┼───────────────────────────────
   1 │     1  a         100  tech
   2 │     2  b         200  energy
```
"""
function lookupjoin(table; key = nothing, unmatched::Symbol = :missing,
    leftprefix = nothing, rightprefix = nothing)
    keycols = tokeycolumns(key)
    isempty(keycols) && throw(ArgumentError("lookupjoin requires a key"))
    allunique(keycols) ||
        throw(ArgumentError("lookupjoin key columns must be unique"))
    mode = unmatchedmode(unmatched)
    Tables.istable(table) || throw(
        ArgumentError("lookupjoin table must be a Tables.jl table, got $(typeof(table))"),
    )
    # Owned 1-based vectors: the index is built now, so a caller mutating the
    # table later must not be able to desynchronize it.
    cols = map(collect, Tables.columntable(table))
    colnames = keys(cols)
    :time in colnames && throw(
        ArgumentError(
            "lookupjoin table has a :time column; a lookup table is timeless \
            — drop that column, or use asofjoin to join on time",
        ),
    )
    for k in keycols
        k in colnames ||
            throw(ArgumentError("lookupjoin key column $(repr(k)) not found in the table"))
    end
    lp = normprefix(leftprefix)
    rp = normprefix(rightprefix)
    valuenames = Tuple(n for n in colnames if !(n in keycols))
    rightnames = Symbol[prefixed(rp, n) for n in valuenames]
    for n in rightnames
        (n === :time || n in keycols) && throw(
            ArgumentError(
                "lookupjoin output column $(repr(n)) appears more than once; use \
                `leftprefix`/`rightprefix` to disambiguate",
            ),
        )
    end
    cfg = buildlookup(cols, Val(Tuple(keycols)), Val(valuenames), keycols, mode,
        lp, rightnames)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> lookupchunk(cfg, c), p.run(ctx))
        end
    end
end
lookupjoin(p::CausalPipeline, table; kwargs...) = lookupjoin(table; kwargs...)(p)

# What an unmatched row does, as a type, so the assembly dispatches on it
# statically rather than branching on a Symbol per chunk.
struct MissingOnUnmatched end
struct ErrorOnUnmatched end
struct DropUnmatched end

function unmatchedmode(s::Symbol)
    s === :missing && return MissingOnUnmatched()
    s === :error && return ErrorOnUnmatched()
    s === :drop && return DropUnmatched()
    throw(
        ArgumentError(
            "lookupjoin unmatched must be :missing, :error or :drop, got $(repr(s))",
        ),
    )
end

# Everything a run needs, concretely typed: the closures capture this, so each
# chunk's call into `lookupchunk` is statically dispatched on the table's types.
struct LookupJoin{K<:NamedTuple,C<:NamedTuple,KN,M}
    index::Dict{K,Int}           # key => row of the table
    values::C                    # the table's non-key columns, owned
    keycols::Vector{Symbol}
    keynames::Val{KN}
    mode::M
    leftprefix::Union{Nothing,String}
    rightnames::Vector{Symbol}   # the value columns' output names, prefixed
end

# Function barrier over the table's typed columns. The key type is the table's,
# as a store key is the right side's in asofjoin; `get!` stores the first row of
# each key, so a returned row other than `i` is a duplicate.
function buildlookup(cols::NamedTuple, keynames::Val{KN}, ::Val{VN},
    keycols::Vector{Symbol}, mode, leftprefix, rightnames) where {KN,VN}
    K = NamedTuple{KN,Tuple{map(k -> eltype(getproperty(cols, k)), KN)...}}
    n = length(getproperty(cols, first(KN)))
    index = Dict{K,Int}()
    sizehint!(index, n)
    for i in 1:n
        k = keyat(cols, i, keynames)
        get!(index, k, i) == i || throwduplicate(k)
    end
    values = NamedTuple{VN}(cols)
    return LookupJoin(index, values, keycols, keynames, mode, leftprefix,
        rightnames)
end

@noinline throwduplicate(k) =
    throw(ArgumentError("lookupjoin table has more than one row with key $k"))

function lookupchunk(cfg::LookupJoin, c::DataFrame)
    checkkeycolumns(cfg.keycols, c, "lookupjoin")
    checklookupnames(cfg, c)
    rows = Vector{Int}(undef, nrow(c))
    nt = Tables.columntable(c)
    nmatched = lookuprows!(rows, cfg.index, nt, cfg.keynames)
    return assemblelookup(cfg.mode, cfg, c, nt, rows, nmatched)
end

# The input's columns after the leftprefix rename, against the time column, the
# keys and the table's output names. O(ncols) per chunk, which keeps the
# transform free of per-run state.
function checklookupnames(cfg::LookupJoin, c::DataFrame)
    for n in propertynames(c)
        (n === :time || n in cfg.keycols) && continue
        m = prefixed(cfg.leftprefix, n)
        (m === :time || m in cfg.keycols || m in cfg.rightnames) && throw(
            ArgumentError(
                "lookupjoin output column $(repr(m)) appears more than once; use \
                `leftprefix`/`rightprefix` to disambiguate",
            ),
        )
    end
    return nothing
end

# --- lookup kernel ---------------------------------------------------------
#
# Each input row's table row, or 0 where its key is absent; returns the number
# matched. Only Ints are stored, so a String key costs nothing per row. The
# input's key may be a different but `isequal` type from the table's: a Dict
# lookup hashes and compares without converting (the KeySet rule).
function lookuprows!(rows::Vector{Int}, index::Dict{K,Int}, nt::NamedTuple,
    keynames::Val{KN}) where {K,KN}
    nmatched = 0
    @inbounds for i in eachindex(rows)
        j = get(index, keyat(nt, i, keynames), 0)
        rows[i] = j
        nmatched += j > 0
    end
    return nmatched
end

# --- output assembly -------------------------------------------------------

function assemblelookup(::MissingOnUnmatched, cfg::LookupJoin, c::DataFrame,
    nt::NamedTuple, rows::Vector{Int}, nmatched::Int)
    return appendlookup(cfg, c, map(col -> gathermissing(col, rows), cfg.values))
end

function assemblelookup(::ErrorOnUnmatched, cfg::LookupJoin, c::DataFrame,
    nt::NamedTuple, rows::Vector{Int}, nmatched::Int)
    nmatched < length(rows) &&
        throwunmatched(keyat(nt, findfirst(==(0), rows), cfg.keynames))
    return appendlookup(cfg, c, map(col -> col[rows], cfg.values))
end

function assemblelookup(::DropUnmatched, cfg::LookupJoin, c::DataFrame,
    nt::NamedTuple, rows::Vector{Int}, nmatched::Int)
    nmatched == 0 && return nothing
    if nmatched < length(rows)
        c = c[findall(>(0), rows), :]
        filter!(>(0), rows)
    end
    return appendlookup(cfg, c, map(col -> col[rows], cfg.values))
end

@noinline throwunmatched(k) = throw(
    ArgumentError(
        "lookupjoin key $k is not in the table; pass `unmatched = :missing` \
        or `unmatched = :drop` to allow it",
    ),
)

# Function barrier: the output column is typed from the table's column, and
# every slot is written, so the undef vector is never read uninitialized.
function gathermissing(col::Vector{T}, rows::Vector{Int}) where {T}
    out = Vector{Union{Missing,T}}(undef, length(rows))
    @inbounds for i in eachindex(rows)
        j = rows[i]
        out[i] = j > 0 ? col[j] : missing
    end
    return out
end

# The chunk is owned, so its columns are adopted and its index renamed in place.
# A table of keys alone appends nothing (and under :drop is a semi-join).
function appendlookup(cfg::LookupJoin, c::DataFrame, gathered::NamedTuple)
    out = prefixleft!(cfg.leftprefix, cfg.keycols, c)
    isempty(cfg.rightnames) && return out
    rdf = DataFrame(collect(AbstractVector, values(gathered)), cfg.rightnames;
        copycols = false)
    return hcat(out, rdf; copycols = false)
end
