# Row functions (filterrows' pred, addcolumns' f) receive the concretely
# typed rows of a Tables.columntable, iterated behind a function barrier —
# never DataFrameRows, whose column accesses are type-unstable.

"""
    filterrows(pred) -> (CausalPipeline -> CausalPipeline)
    filterrows(p::CausalPipeline, pred) -> CausalPipeline

A transform keeping the rows for which `pred(row)` is `true`.

# Arguments
- `pred`: a function `row -> Bool`. `row` supports `row.name` and `row[:name]`,
  including `row.time`.
"""
function filterrows(pred)
    return pipetransform() do upstream, ctx
        return chunkmap(c -> filterchunk(pred, c), upstream)
    end
end
filterrows(p::CausalPipeline, pred) = filterrows(pred)(p)

function filterchunk(pred, c::DataFrame)
    mask, kept = rowmask(pred, Tables.columntable(c))
    # The chunk is owned, so a filter that keeps every row needs no copy.
    return kept == length(mask) ? c : c[mask, :]
end

# Function barrier over concretely typed rows. Counting kept rows in the same
# pass saves a second walk for the "everything survived" test.
function rowmask(pred, nt::NamedTuple)
    mask = Vector{Bool}(undef, length(nt.time))
    kept = 0
    for (i, row) in enumerate(Tables.rows(nt))
        keep = pred(row)::Bool
        @inbounds mask[i] = keep
        kept += keep
    end
    return mask, kept
end

"""
    addcolumns(f) -> (CausalPipeline -> CausalPipeline)
    addcolumns(p::CausalPipeline, f) -> CausalPipeline

A transform appending columns computed from each row.

# Arguments
- `f`: a function `row -> NamedTuple`, where the `NamedTuple` maps each new
  column name to its value in that row. `row` is as for [`filterrows`](@ref).
  Returning anything but a `NamedTuple`, or a name already in the input
  (including `time`), is an `ArgumentError`.

```jldoctest
p = readtable(DataFrame(time = [1, 2], bid = [10.0, 10.5], ask = [10.2, 10.7]))
DataFrame(load(Context(0, 10), p |> addcolumns(r -> (; mid = (r.bid + r.ask) / 2))))

# output

2×4 DataFrame
 Row │ time   bid      ask      mid
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     10.0     10.2     10.1
   2 │     2     10.5     10.7     10.6
```
"""
function addcolumns(f)
    return pipetransform() do upstream, ctx
        return chunkmap(c -> addchunk(f, c), upstream)
    end
end
addcolumns(p::CausalPipeline, f) = addcolumns(f)(p)

function addchunk(f, c::DataFrame)
    vals = rowvalues(f, Tables.columntable(c))
    first(vals) isa NamedTuple || throw(
        ArgumentError(
            "addcolumns function must return a NamedTuple, got $(typeof(first(vals)))"),
    )
    :time in keys(first(vals)) && throw(ArgumentError(
        "addcolumns function may not return a time column"))
    checknewcolumns(c, keys(first(vals)), "addcolumns")
    # The chunk is owned, so its columns can be adopted rather than copied.
    return hcat(c, rowsdataframe(vals); copycols = false)
end

# Function barrier: over concretely typed rows the comprehension infers a
# concrete NamedTuple eltype, so DataFrame builds typed columns directly.
rowvalues(f, nt::NamedTuple) = [f(row) for row in Tables.rows(nt)]
