# Model fitting and prediction through MLJ. Like parquet.jl, this file names no
# MLJ type: fitting, prediction and fitresult persistence go through five hooks
# whose generic fallbacks live here and whose methods for MLJModelInterface
# models live in ext/CausalFramesMLJModelInterfaceExt.jl, so a user who never
# loads an MLJ package pays nothing. The summarizer (`FitModel`) and the value
# type it emits (`FittedModel`) are in summarizers.jl; the operators over
# tables of fitted models are here.

const MLJHINT = "model fitting needs MLJModelInterface: `using MLJ`, or any \
    MLJ model package"

mljloaded() =
    Base.get_extension(@__MODULE__, :CausalFramesMLJModelInterfaceExt) !== nothing

# --- the extension hooks ---------------------------------------------------

# Whether `m` is a model the extension can fit. Consulted eagerly by FitModel's
# constructor, so a missing extension is reported where the user typed it.
ismodel(::Any) = false

# Fit `model` to the column table `X` and response vector `y`, returning
# `(fitresult, report)`.
fitmodel(model, verbosity::Int, X, y) = throw(ArgumentError(MLJHINT))

# Apply the operation `op` (:predict, :predict_mean, :predict_mode or
# :predict_median) of a fitted model to the column table `X`.
predictmodel(model, fitresult, op::Symbol, X) = throw(ArgumentError(MLJHINT))

# A persistent form of a fitresult and its inverse (MLJ's `save`/`restore`),
# the identity unless a model wraps a resource that cannot be serialized as is.
savefitresult(model, fitresult) = fitresult
restorefitresult(model, stored) = stored

# Serializing a FittedModel — directly, or as a column cell under writejls —
# routes its fitresult through the save/restore hooks. The type is our own, so
# this extends Serialization for it rather than pirating anything.
function Serialization.serialize(s::Serialization.AbstractSerializer,
    fm::FittedModel)
    Serialization.serialize_type(s, typeof(fm))
    serialize(s, fm.model)
    serialize(s, savefitresult(fm.model, fm.fitresult))
    serialize(s, fm.report)
    return nothing
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
    ::Type{FittedModel{P,M}}) where {P,M}
    model = deserialize(s)
    stored = deserialize(s)
    report = deserialize(s)
    return FittedModel{P,M}(model, restorefitresult(model, stored), report)
end

# --- applymodels -----------------------------------------------------------

const PREDICTOPS = (:predict, :predict_mean, :predict_mode, :predict_median)

"""
    applymodels(models::CausalPipeline; column = :model, key = nothing,
                tolerance = nothing, strict = false, name = :prediction,
                operation = :predict) -> (CausalPipeline -> CausalPipeline)
    applymodels(p::CausalPipeline, models::CausalPipeline; ...) -> CausalPipeline

A transform appending each row's prediction from the latest model at or before
its time, matched as by [`asofjoin`](@ref). The prediction column is
`Union{Missing, T}`: `missing` where no model matches or the matched model is
`missing`. Rows sharing a model are predicted in one call per chunk. Requires
MLJModelInterface (`using MLJ`).

# Arguments
- `models`: a pipeline with a column of [`FittedModel`](@ref)s, such as
  [`FitModel`](@ref) under a summarizing transform, or a file of them read with
  [`readjls`](@ref). If it produces no rows, every prediction is `missing`.

# Keywords
- `column = :model`: the column of `models` holding the models.
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`, present in both pipelines; each row uses its own key's latest
  model.
- `tolerance = nothing`: the maximum age of a model, as for `asofjoin`. It
  widens the models' context to `[start - tolerance, stop)`, which is how a
  model fit in an earlier window reaches a later one.
- `strict = false`: use only models strictly before the row.
- `name = :prediction`: the output column.
- `operation = :predict`: the MLJ operation, one of `:predict` (a probabilistic
  model then gives distributions), `:predict_mean`, `:predict_mode` or
  `:predict_median`. It must return a vector; a multi-target table is an
  `ArgumentError`.
"""
function applymodels(models::CausalPipeline; column::Symbol = :model,
    key = nothing, tolerance = nothing, strict::Bool = false,
    name::Symbol = :prediction, operation::Symbol = :predict)
    keycols = keycolumns(key, "applymodels")
    (column === :time || column in keycols) && throw(
        ArgumentError(
            "applymodels column $(repr(column)) may not be :time or a key column"),
    )
    (name === :time || name in keycols) && throw(
        ArgumentError(
            "applymodels output column $(repr(name)) may not be :time or a key column"),
    )
    operation in PREDICTOPS || throw(
        ArgumentError(
            "applymodels operation must be one of $PREDICTOPS, got $(repr(operation))"),
    )
    # Only the model and key columns are carried into the as-of store. A
    # predicate selector, unlike a name, does not fail on an absent column, so
    # a missing model column is reported below under this operator's name.
    right = models |> selectcolumns(n -> Symbol(n) === column || Symbol(n) in keycols)
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            cfg = JoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                strict ? (<) : (<=), Backward(), nothing, nothing, nothing,
                "applymodels")
            js = JoinState(right.run(widenstart(ctx, tolerance, "applymodels tolerance")))
            return chunkmap(c -> predictchunk!(js, cfg, column, name, operation, c),
                p.run(ctx))
        end
    end
end
applymodels(p::CausalPipeline, models::CausalPipeline; kwargs...) =
    applymodels(models; kwargs...)(p)

# joinchunk!'s driver, with the prediction column in place of the right
# table's columns. The as-of match itself is join.jl's, unchanged.
function predictchunk!(js::JoinState, cfg::JoinConfig, column::Symbol,
    name::Symbol, op::Symbol, c::DataFrame)
    if !js.leftchecked
        checkkeycolumns(cfg.keycols, c, cfg.op, "the left input")
        String(name) in names(c) && throw(
            ArgumentError(
                "applymodels output column $(repr(name)) collides with an existing column",
            ),
        )
        js.leftchecked = true
    end
    js.rnt === nothing && !js.right.done && pullright!(js, cfg)
    if js.passthrough
        c[!, name] = fill(missing, nrow(c))
        return c
    end
    if !js.checked
        column in js.rvaluenames || throw(
            ArgumentError(
                "applymodels: the models pipeline has no column named $(repr(column))"),
        )
        js.checked = true
    end
    nt = Tables.columntable(c)
    matchchunk!(js, cfg, nt)
    c[!, name] = predictcolumn(js.matches, js.found, Val(column), nt, op)
    return c
end

# Function barrier: with the match buffer's row type known, the model cells are
# concretely typed and grouping them costs one IdDict insert per row. Each
# distinct model is then applied once to a view of its rows — the only dynamic
# calls, one per model per chunk — and the results are scattered into a single
# column whose element type is the promotion of every group's.
function predictcolumn(matches::Vector{V}, found::Vector{Bool}, ::Val{C},
    nt::NamedTuple, op::Symbol) where {V,C}
    FM = nonmissingtype(fieldtype(V, C))
    groups = IdDict{FM,Vector{Int}}()
    for i in eachindex(matches)
        @inbounds(found[i]) || continue
        m = getproperty(@inbounds(matches[i]), C)
        ismissing(m) && continue
        m isa FittedModel || throw(
            ArgumentError(
                "applymodels: column $C holds a $(typeof(m)), not a FittedModel"),
        )
        push!(get!(Vector{Int}, groups, m), i)
    end
    preds = Pair{Vector{Int},AbstractVector}[
        idx => predictgroup(m, nt, idx, op) for (m, idx) in groups]
    E = mapreduce(pr -> eltype(last(pr)), promote_type, preds; init = Union{})
    out = Vector{Union{Missing,E}}(missing, length(matches))
    for (idx, pr) in preds
        scatter!(out, idx, pr)
    end
    return out
end

function predictgroup(fm::FittedModel{P}, nt::NamedTuple, idx::Vector{Int},
    op::Symbol) where {P}
    for p in P
        hasproperty(nt, p) || throw(
            ArgumentError(
                "applymodels: predictor column $(repr(p)) not found in the input"),
        )
    end
    X = NamedTuple{P}(map(p -> view(getproperty(nt, p), idx), P))
    pr = predictmodel(fm.model, fm.fitresult, op, X)
    pr isa AbstractVector || throw(ArgumentError("applymodels: $op returned a \
        $(typeof(pr)), not a vector; multi-target models are not supported"))
    length(pr) == length(idx) || throw(ArgumentError("applymodels: $op \
        returned $(length(pr)) predictions for $(length(idx)) rows"))
    return pr
end

# A function barrier on the prediction vector's concrete type.
function scatter!(out::Vector, idx::Vector{Int}, pr::AbstractVector)
    for (j, i) in enumerate(idx)
        @inbounds out[i] = pr[j]
    end
    return out
end

# --- addpredictions --------------------------------------------------------

"""
    addpredictions(clock, lookback, model, predictors, response; key = nothing,
                   name = :prediction, operation = :predict,
                   verbosity = 0) -> (CausalPipeline -> CausalPipeline)
    addpredictions(p::CausalPipeline, clock, lookback, model, predictors,
                   response; ...) -> CausalPipeline

A transform appending predictions from a model refit on a trailing window. At
each clock tick `τ`, `model` is fit to the rows in `[τ - lookback, τ)` (per key
with `key`); each row is then predicted by the model of the latest tick at or
before it, so training rows always precede the rows they predict. Rows before
the first tick, or whose latest window was empty, get `missing`. It is
[`summarizewindows`](@ref) over [`FitModel`](@ref) feeding
[`applymodels`](@ref), and the input runs twice.

# Arguments
- `clock`: a pipeline whose `:time` column gives the refit times, such as
  [`clock`](@ref).
- `lookback`: the training window length, as for `summarizewindows`.
- `model`, `predictors`, `response`: as for [`FitModel`](@ref). The response
  must be known at its row's time; one built with
  [`Acausal.lead`](@ref CausalFrames.Acausal.lead) leaks the future into
  training.

# Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`; fit and apply one model per key.
- `name = :prediction`, `operation = :predict`: as for [`applymodels`](@ref).
- `verbosity = 0`: passed to the model's `fit`.
"""
function addpredictions(clk::CausalPipeline, lookback, model, predictors,
    response::Symbol; key = nothing, name::Symbol = :prediction,
    operation::Symbol = :predict, verbosity::Integer = 0)
    fitter = FitModel(model, predictors, response; name = :model, verbosity)
    fits = summarizewindows(clk, lookback, fitter; key)
    return function (p::CausalPipeline)
        return p |> applymodels(p |> fits; column = :model, key, strict = false,
            name, operation)
    end
end
addpredictions(p::CausalPipeline, clk::CausalPipeline, lookback, model,
    predictors, response::Symbol; kwargs...) =
    addpredictions(clk, lookback, model, predictors, response; kwargs...)(p)

# --- modelreports ----------------------------------------------------------

"""
    modelreports(; column = :model, name = :report) -> (CausalPipeline -> CausalPipeline)
    modelreports(p::CausalPipeline; column = :model, name = :report) -> CausalPipeline

A transform, over a pipeline of models, replacing each [`FittedModel`](@ref)
with its fit report: what MLJ's `report(mach)` returns after `fit!` (`nothing`
for a model with nothing to report). `missing` stays `missing`; other columns
pass through. Extract fields with [`addcolumns`](@ref), e.g.
`addcolumns(r -> (; n = r.report.n))`.

# Keywords
- `column = :model`: the column holding the models. May not be `:time`.
- `name = :report`: the output column, replacing `column`. May not be `:time`.
"""
function modelreports(; column::Symbol = :model, name::Symbol = :report)
    column === :time &&
        throw(ArgumentError("modelreports column may not be :time"))
    name === :time &&
        throw(ArgumentError("modelreports output column may not be named :time"))
    return function (p::CausalPipeline)
        return CausalPipeline() do ctx::Context
            return chunkmap(c -> reportchunk!(c, column, name), p.run(ctx))
        end
    end
end
modelreports(p::CausalPipeline; kwargs...) = modelreports(; kwargs...)(p)

# The chunk is owned, so the model column is replaced in its own position and
# renamed; the column vector itself is new, never mutated in place.
function reportchunk!(c::DataFrame, column::Symbol, name::Symbol)
    hasproperty(c, column) ||
        throw(ArgumentError("modelreports: no column named $(repr(column))"))
    name !== column && hasproperty(c, name) &&
        throw(
            ArgumentError(
                "modelreports output column $(repr(name)) collides with an existing column",
            ),
        )
    # `map` narrows the element type from the reports themselves, the field
    # being untyped: Union{Missing, R} for a stream of R-typed reports.
    c[!, column] = map(fitreport, c[!, column])
    name === column || rename!(c, column => name)
    return c
end

fitreport(::Missing) = missing
fitreport(fm::FittedModel) = fm.report
fitreport(x) = throw(
    ArgumentError(
        "modelreports: the model column holds a $(typeof(x)), not a FittedModel"),
)
