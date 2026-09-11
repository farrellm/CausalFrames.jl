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

A transform appending model predictions. `models` is a pipeline whose `column`
holds [`FittedModel`](@ref)s — the output of [`FitModel`](@ref) under any
summarization transform, or a file of them read back by [`readjls`](@ref).
Each row is matched to the most recent model row whose time is not after its
own (`strict = true`: strictly before) — [`asofjoin`](@ref)'s rule, including
its `key` and `tolerance` — and that model's prediction from the row's
predictor columns is appended as the column `name`, of element type
`Union{Missing, T}`. It is `missing` where no model row qualifies or the
matched cell is itself `missing` (an empty window's summary). A `models` stream
with no rows at all appends an all-`missing` column.

Rows sharing a model are predicted together, in one call per distinct model per
chunk. `operation` chooses the MLJ operation: `:predict` (the default; a
probabilistic model then yields distributions), `:predict_mean`,
`:predict_mode` or `:predict_median`. The prediction must be a vector — a
multi-target model's table is an `ArgumentError`.

With `key` the models pipeline must carry the key columns and each row uses its
own key's latest model. `tolerance` widens the models' context to
`[start - tolerance, stop)`, which is how models fit in an earlier window are
applied to a later one; see the manual's Recipes page. Needs MLJModelInterface
loaded, as `FitModel` does.

The curried form composes with `|>`; the uncurried form applies directly, so
`applymodels(p, models; ...)` is equivalent to `p |> applymodels(models; ...)`.
"""
function applymodels(models::CausalPipeline; column::Symbol = :model,
    key = nothing, tolerance = nothing, strict::Bool = false,
    name::Symbol = :prediction, operation::Symbol = :predict)
    keycols = tokeycolumns(key)
    allunique(keycols) ||
        throw(ArgumentError("applymodels key columns must be unique"))
    :time in keycols && throw(
        ArgumentError(
            "time is the as-of dimension and may not be an applymodels key"),
    )
    (column === :time || column in keycols) && throw(ArgumentError(
        "applymodels column $column may not be time or a key column"))
    (name === :time || name in keycols) && throw(
        ArgumentError(
            "applymodels output column $name may not be time or a key column"),
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
            cfg = AsofJoinConfig(keycols, Val(Tuple(keycols)), tolerance,
                strict ? (<) : (<=), nothing, nothing, nothing, "applymodels")
            js = AsofJoinState(right.run(rightcontext(ctx, tolerance,
                "applymodels")))
            return chunkmap(c -> predictchunk!(js, cfg, column, name, operation, c),
                p.run(ctx))
        end
    end
end
applymodels(p::CausalPipeline, models::CausalPipeline; kwargs...) =
    applymodels(models; kwargs...)(p)

# joinchunk!'s driver, with the prediction column in place of the right
# table's columns. The as-of match itself is join.jl's, unchanged.
function predictchunk!(js::AsofJoinState, cfg::AsofJoinConfig, column::Symbol,
    name::Symbol, op::Symbol, c::DataFrame)
    if !js.leftchecked
        checkkeys(cfg.keycols, c, "left", cfg.op)
        String(name) in names(c) && throw(
            ArgumentError(
                "applymodels output column $name collides with an existing column"),
        )
        js.leftchecked = true
    end
    js.rnt === nothing && !js.rdone && pullright!(js, cfg)
    if js.passthrough
        c[!, name] = fill(missing, nrow(c))
        return c
    end
    if !js.checked
        column in js.rvaluenames || throw(ArgumentError(
            "applymodels: the models pipeline has no column $column"))
        js.checked = true
    end
    nt = Tables.columntable(c)
    resize!(js.matches, nrow(c))
    resize!(js.found, nrow(c))
    fill!(js.found, false)
    i = 1
    while true
        i, js.rpos, needpull = joinsegment!(js.matches, js.found, js.index,
            js.slots, nt, i, js.rnt, js.rpos, js.rdone, cfg.keynames,
            cfg.before, cfg.tolerance)
        needpull || break
        pullright!(js, cfg)
    end
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
        hasproperty(nt, p) || throw(ArgumentError(
            "applymodels: predictor column $p not found in the input"))
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

A transform appending the predictions of an MLJ `model` refit on a rolling
window. At every tick `τ` of the `clock` pipeline the model is fit, regressing
`response` on `predictors`, over the rows with time in `[τ - lookback, τ)` —
one model per key present there when `key` is given — and each row at time `t`
is then predicted by the model from the latest tick `τ <= t`. Every row a model
was trained on is therefore strictly earlier than every row it predicts.

This is [`summarizewindows`](@ref) over a [`FitModel`](@ref) feeding
[`applymodels`](@ref), and it inherits their semantics: rows before the first
tick, a key with no rows in the latest window, and a window with no rows all
get `missing`; `name` and `operation` are `applymodels`'; `verbosity` is
passed to the model's `fit`. The pipeline runs twice, once to fit and once to
predict, as a self-join does.

Causality needs one thing from the data: `response` must be observable at its
row's time. A response built by looking ahead (with `CausalFrames.Acausal.lead`,
say) trains every model on values that were not yet known at its tick.
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

A row-wise transform over a pipeline of models, replacing its `column` of
[`FittedModel`](@ref)s with a column `name` holding each model's fit report —
the diagnostics MLJ's `fit` returned alongside the fitresult, which MLJ's
`report(mach)` gives for a machine. A `missing` cell stays `missing`. The time
column, keys and every other column pass through, so the result is a pipeline
of diagnostics per tick (and key).

The report stays one column rather than being spread into its fields: a
chunk's columns must be fixed before its rows go out, and a report's fields are
unknown until a model has been fit — a stream opening on empty windows would
have nothing to name them from. Extract fields with [`addcolumns`](@ref), e.g.
`addcolumns(r -> (; n = r.report.n))`.

The curried form composes with `|>`; the uncurried form applies directly, so
`modelreports(p; ...)` is equivalent to `p |> modelreports(; ...)`.
"""
function modelreports(; column::Symbol = :model, name::Symbol = :report)
    column === :time &&
        throw(ArgumentError("modelreports column may not be time"))
    name === :time &&
        throw(ArgumentError("modelreports output column may not be named time"))
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
                "modelreports output column $name collides with an existing column"),
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
