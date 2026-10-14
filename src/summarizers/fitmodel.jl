"""
    FittedModel{P,M}

A fitted MLJ model, as [`FitModel`](@ref) emits it. It holds the `model`
(hyperparameters, of type `M`), the `fitresult` needed for prediction, and the
fit `report` (see [`modelreports`](@ref)). `P` is the tuple of predictor names,
which [`applymodels`](@ref) predicts from. Serialization, as in
[`writejls`](@ref), goes through MLJ's `save`/`restore`.
"""
struct FittedModel{P,M}
    model::M
    fitresult::Any
    report::Any
end

# Compact, so a frame of models prints as a table.
Base.show(io::IO, fm::FittedModel{P}) where {P} =
    print(io, "FittedModel(", nameof(typeof(fm.model)), ", ", P, ")")

"""
    FitModel(model, predictors, response::Symbol; name = :model,
             verbosity = 0) -> Summarizer

Fits an MLJ model to the rows it summarizes, emitting a [`FittedModel`](@ref)
(`missing` for no rows). Requires MLJModelInterface: `using MLJ`, or an MLJ
model package (most models also need MLJBase, which `using MLJ` loads).

# Arguments
- `model`: an `MLJModelInterface.Model`; anything else is an `ArgumentError`.
- `predictors`: a column name, or a non-empty collection of distinct names.
  They reach the model as a column table of their own element types, without
  scientific-type coercion (coerce upstream with [`addcolumns`](@ref)), and
  `missing` is passed through. A row term (see [`ColumnSpec`](@ref)) is an
  `ArgumentError`: [`applymodels`](@ref) predicts from another stream, which
  would lack its column.
- `response`: the response column, which may not be a predictor.

# Keywords
- `name = :model`: the output column.
- `verbosity = 0`: passed to the model's `fit`.

Rows are buffered and the model is fit each time a summary is emitted: once
under [`summarize`](@ref), per interval under [`intervalize`](@ref), per tick
and key under [`summarizewindows`](@ref), and after every row (quadratic)
under [`addsummarycolumns`](@ref). Windows re-fold for it, as it neither
combines nor inverts. Apply the models with [`applymodels`](@ref), or fit and
apply in one step with [`addpredictions`](@ref).
"""
struct FitModel{N,P,Y,M} <: Summarizer
    model::M
    verbosity::Int
end

function FitModel(model, predictors, response::Symbol; name::Symbol = :model,
    verbosity::Integer = 0)
    ismodel(model) || throw(
        ArgumentError(
            mljloaded() ?
            "FitModel model must be an MLJ model, got a $(typeof(model))" : MLJHINT,
        ),
    )
    # `applymodels` predicts from another stream, where a row term's virtual
    # column does not exist.
    (predictors isa Pair || any(p -> p isa Pair, predictors)) && throw(
        ArgumentError(
            "FitModel predictors must be column names, got a row term"),
    )
    ps = Tuple(Symbol(p) for p in predictors)
    isempty(ps) && throw(ArgumentError("FitModel requires at least one predictor"))
    allunique(ps) || throw(ArgumentError(
        "FitModel predictors must be unique, got $ps"))
    response in ps && throw(ArgumentError(
        "FitModel response $(repr(response)) is also a predictor"))
    return FitModel{name,ps,response,typeof(model)}(model, Int(verbosity))
end
# A lone name; a string is one name, not a collection of characters.
FitModel(model, predictor::Union{Symbol,AbstractString}, response::Symbol;
    kwargs...) = FitModel(model, (Symbol(predictor),), response; kwargs...)

# The folded rows: a concretely typed vector per predictor and one for the
# response. `fresh!` empties them keeping capacity, which is safe only because
# `value` fits on copies.
mutable struct FitModelState{N,P,Y,M,C<:NamedTuple,V<:AbstractVector} <:
               SummarizerState
    const model::M
    const verbosity::Int
    cols::C
    y::V
end

emptyvalue(::FitModel{N}) where {N} = NamedTuple{(N,)}((missing,))
function fresh(s::FitModel{N,P,Y,M}, intypes::NamedTuple) where {N,P,Y,M}
    cols = NamedTuple{P}(map(p -> Vector{intypes[p]}(), P))
    y = Vector{intypes[Y]}()
    return FitModelState{N,P,Y,M,typeof(cols),typeof(y)}(s.model, s.verbosity,
        cols, y)
end
fresh(st::FitModelState{N,P,Y,M,C,V}) where {N,P,Y,M,C,V} =
    FitModelState{N,P,Y,M,C,V}(st.model, st.verbosity,
        map(v -> similar(v, 0), st.cols), similar(st.y, 0))
@inline function fresh!(st::FitModelState)
    foreach(empty!, values(st.cols))
    empty!(st.y)
    return st
end
# `P` is a type parameter, so the map unrolls into one typed push per column.
# `map`, not `foreach`: Julia 1.10's two-tuple `foreach` goes through `zip`
# and dispatches per column. When 1.10 support is dropped, use `foreach`.
@inline function update!(st::FitModelState{N,P,Y}, row) where {N,P,Y}
    map(push!, values(st.cols), map(p -> getproperty(row, p), P))
    push!(st.y, getproperty(row, Y))
    return nothing
end
# The fit returns Any, but the value's type is built from the type parameters,
# so the output column is concrete.
function value(st::FitModelState{N,P,Y,M}) where {N,P,Y,M}
    fitresult, report = fitmodel(st.model, st.verbosity, map(copy, st.cols),
        copy(st.y))
    return NamedTuple{(N,),Tuple{FittedModel{P,M}}}((
        FittedModel{P,M}(st.model,
            fitresult, report),
    ))
end
function widenstate(st::FitModelState{N,P,Y,M},
    intypes::NamedTuple) where {N,P,Y,M}
    cols = NamedTuple{P}(map(p -> convert(Vector{intypes[p]}, st.cols[p]), P))
    y = convert(Vector{intypes[Y]}, st.y)
    return FitModelState{N,P,Y,M,typeof(cols),typeof(y)}(st.model, st.verbosity,
        cols, y)
end
