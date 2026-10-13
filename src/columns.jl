# The column projections and `reordercolumns`, over one selector vocabulary
# and a per-run, schema-keyed resolution memo.

"""
    selectcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    selectcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform keeping only the selected columns, in their input order. `:time` is
always kept.

# Arguments
- `selectors`: at least one column selector: a name (`Symbol` or `String`),
  a `Regex` matched against names, a predicate called with the name as a
  `String`, or a collection of these. A column is kept if any selector matches
  it. A name the data lacks is an `ArgumentError`; a `Regex` or predicate
  matching nothing is not.

```jldoctest
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> selectcolumns(:venue, r"^px_", startswith("qty"))
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "px_bid"
 "px_ask"
 "qty_bid"
 "venue"
```
"""
function selectcolumns(selectors...)
    checkselectors(selectors, "selectcolumns", false)
    return columntransform(cols -> keptcolumns(selectors, cols, true,
        "selectcolumns"))
end
selectcolumns(p::CausalPipeline, selectors...) = selectcolumns(selectors...)(p)

"""
    dropcolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    dropcolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform removing the selected columns and keeping the rest, in their input
order.

# Arguments
- `selectors`: at least one column selector, as for
  [`selectcolumns`](@ref). A column is dropped if any selector matches it.
  `:time` is never dropped: a `Regex` or predicate matching it is ignored, and
  naming it is an `ArgumentError`, as is naming a column the data lacks.
"""
function dropcolumns(selectors...)
    checkselectors(selectors, "dropcolumns", true)
    return columntransform(cols -> keptcolumns(selectors, cols, false,
        "dropcolumns"))
end
dropcolumns(p::CausalPipeline, selectors...) = dropcolumns(selectors...)(p)

"""
    reordercolumns(selectors...) -> (CausalPipeline -> CausalPipeline)
    reordercolumns(p::CausalPipeline, selectors...) -> CausalPipeline

A transform moving the selected columns to the front, after `:time`, in the
order of the selectors. The other columns follow in their input order.

# Arguments
- `selectors`: at least one column selector, as for
  [`selectcolumns`](@ref). Nested collections are flattened in place. A `Regex`
  or predicate contributes its matches in input order, and a column matched
  twice goes where it was first matched. `:time` always stays first: a `Regex`
  or predicate matching it is ignored, and naming it is an `ArgumentError`, as
  is naming a column the data lacks.

```jldoctest
df = DataFrame(time = [1], px_bid = [10.0], px_ask = [10.2], qty_bid = [5], venue = ["X"])
p = readtable(df) |> reordercolumns(:venue, r"^qty_")   # time, venue, qty_bid, then the rest
names(load(Context(0, 10), p))

# output

5-element Vector{String}:
 "time"
 "venue"
 "qty_bid"
 "px_bid"
 "px_ask"
```
"""
function reordercolumns(selectors...)
    checkselectors(selectors, "reordercolumns", false)
    foreachliteral(selectors) do n
        n == "time" && throw(
            ArgumentError("reordercolumns: the time column is always first"))
    end
    return columntransform(cols -> orderedcolumns(selectors, cols))
end
reordercolumns(p::CausalPipeline, selectors...) = reordercolumns(selectors...)(p)

# The column transforms' shared shape: a chunkmap indexing each owned chunk by
# `resolve(names)`, the columns to keep in order, or `nothing` to pass it
# through.
function columntransform(resolve)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            memo = SchemaMemo()
            return chunkmap(p.run(ctx)) do c
                keep = memoized!(resolve, memo, c)
                return keep === nothing ? c : c[!, keep]
            end
        end
    end
end

# The resolution, cached against the names it came from. The schema rarely
# changes, so this skips re-running the selectors, while re-resolving when names
# differ keeps validation per-chunk (load and stream don't recheck schemas).
mutable struct SchemaMemo
    lastnames::Union{Nothing,Vector{String}}
    resolved::Union{Nothing,Vector{Symbol}}
end
SchemaMemo() = SchemaMemo(nothing, nothing)

function memoized!(resolve, memo::SchemaMemo, c::DataFrame)
    cols = names(c)
    memo.lastnames == cols && return memo.resolved
    memo.resolved = resolve(cols)
    memo.lastnames = cols
    return memo.resolved
end

# The names to keep, in the chunk's own column order, or `nothing` when every
# column survives.
function keptcolumns(selectors::Tuple, cols::Vector{String}, selecting::Bool,
    opname::String)
    foreachliteral(selectors) do n
        n in cols || throw(
            ArgumentError("$opname: no column named $(repr(Symbol(n)))"))
    end
    keep = Symbol[]
    for n in cols
        (n == "time" || matchescolumn(selectors, n) == selecting) &&
            push!(keep, Symbol(n))
    end
    return length(keep) == length(cols) ? nothing : keep
end

# The column order to impose, or `nothing` when the chunk already has it.
# `placed` tracks columns by position, avoiding hashing.
function orderedcolumns(selectors::Tuple, cols::Vector{String})
    foreachliteral(selectors) do n
        n in cols || throw(
            ArgumentError("reordercolumns: no column named $(repr(Symbol(n)))"))
    end
    order = Symbol[:time]
    placed = falses(length(cols))
    for (i, n) in pairs(cols)
        n == "time" && (placed[i] = true)
    end
    foreachselector(selectors) do s
        for (i, n) in pairs(cols)
            placed[i] && continue
            matchescolumn(s, n) || continue
            push!(order, Symbol(n))
            placed[i] = true
        end
    end
    for (i, n) in pairs(cols)
        placed[i] || push!(order, Symbol(n))
    end
    return all(i -> order[i] === Symbol(cols[i]), eachindex(cols)) ? nothing :
           order
end

# Numbers and Chars iterate as themselves, so the collection fallbacks below
# would recurse forever on them.
const ScalarSelector = Union{Number,Char}

selectorerror(x) = throw(
    ArgumentError("invalid column selector of type $(typeof(x)): expected a \
        name, a Regex, a predicate, or a collection of those"))

checkselector(selectors) =
    applicable(iterate, selectors) || selectorerror(selectors)

# Whether a selector spec matches a column name. Leaves dispatch first, so a
# predicate is never taken for a collection; anything else must be iterable
# and is matched recursively.
matchescolumn(s::Symbol, name::AbstractString) = String(s) == name
matchescolumn(s::AbstractString, name::AbstractString) = String(s) == name
matchescolumn(r::Regex, name::AbstractString) = occursin(r, name)
matchescolumn(f::Function, name::AbstractString) = f(name)::Bool
matchescolumn(x::ScalarSelector, ::AbstractString) = selectorerror(x)
function matchescolumn(selectors, name::AbstractString)
    checkselector(selectors)
    return any(s -> matchescolumn(s, name), selectors)
end

# Walk the name leaves of a selector spec, ignoring regex and predicate ones.
foreachliteral(f, s::Symbol) = (f(String(s)); nothing)
foreachliteral(f, s::AbstractString) = (f(String(s)); nothing)
foreachliteral(::Any, ::Regex) = nothing
foreachliteral(::Any, ::Function) = nothing
foreachliteral(::Any, x::ScalarSelector) = selectorerror(x)
function foreachliteral(f, selectors)
    checkselector(selectors)
    for s in selectors
        foreachliteral(f, s)
    end
    return nothing
end

# Walk every leaf of a selector spec in order, including the regex and
# predicate leaves `foreachliteral` skips.
foreachselector(f, s::Symbol) = (f(s); nothing)
foreachselector(f, s::AbstractString) = (f(s); nothing)
foreachselector(f, r::Regex) = (f(r); nothing)
foreachselector(f, g::Function) = (f(g); nothing)
foreachselector(::Any, x::ScalarSelector) = selectorerror(x)
function foreachselector(f, selectors)
    checkselector(selectors)
    for s in selectors
        foreachselector(f, s)
    end
    return nothing
end

# Eager validation: at least one selector, every leaf usable, and, for
# dropcolumns, no literal `:time`.
function checkselectors(selectors::Tuple, opname::String, dropping::Bool)
    isempty(selectors) &&
        throw(ArgumentError("$opname requires at least one column selector"))
    foreachliteral(selectors) do n
        dropping && n == "time" &&
            throw(ArgumentError("$opname may not drop the time column"))
    end
    return nothing
end
