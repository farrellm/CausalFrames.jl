# Model fitting (MLJ)

Contents: `applymodels`, `addpredictions`, `modelreports`, Fitted models.

With MLJ loaded (`using MLJ`), the `FitModel` summarizer
fits a model inside any summarizing transform, emitting
`FittedModel`s. These operators apply such models and read their
diagnostics.

## `applymodels`

    applymodels(models::CausalPipeline; column = :model, key = nothing,
                tolerance = nothing, strict = false, name = :prediction,
                operation = :predict) -> (CausalPipeline -> CausalPipeline)
    applymodels(p::CausalPipeline, models::CausalPipeline; ...) -> CausalPipeline

A transform appending each row's prediction from the latest model at or before
its time, matched as by `asofjoin`. The prediction column is
`Union{Missing, T}`: `missing` where no model matches or the matched model is
`missing`. Rows sharing a model are predicted in one call per chunk. Requires
MLJModelInterface (`using MLJ`).

#### Arguments
- `models`: a pipeline with a column of `FittedModel`s, such as
  `FitModel` under a summarizing transform, or a file of them read with
  `readjls`. If it produces no rows, every prediction is `missing`.

#### Keywords
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

## `addpredictions`

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
`summarizewindows` over `FitModel` feeding
`applymodels`, and the input runs twice.

#### Arguments
- `clock`: a pipeline whose `:time` column gives the refit times, such as
  `clock`.
- `lookback`: the training window length, as for `summarizewindows`.
- `model`, `predictors`, `response`: as for `FitModel`. The response
  must be known at its row's time; one built with
  `Acausal.lead` leaks the future into
  training.

#### Keywords
- `key = nothing`: a column name or collection of distinct column names other
  than `:time`; fit and apply one model per key.
- `name = :prediction`, `operation = :predict`: as for `applymodels`.
- `verbosity = 0`: passed to the model's `fit`.

## `modelreports`

    modelreports(; column = :model, name = :report) -> (CausalPipeline -> CausalPipeline)
    modelreports(p::CausalPipeline; column = :model, name = :report) -> CausalPipeline

A transform, over a pipeline of models, replacing each `FittedModel`
with its fit report: what MLJ's `report(mach)` returns after `fit!` (`nothing`
for a model with nothing to report). `missing` stays `missing`; other columns
pass through. Extract fields with `addcolumns`, e.g.
`addcolumns(r -> (; n = r.report.n))`.

#### Keywords
- `column = :model`: the column holding the models. May not be `:time`.
- `name = :report`: the output column, replacing `column`. May not be `:time`.

## Fitted models

    FittedModel{P,M}

A fitted MLJ model, as `FitModel` emits it. It holds the `model`
(hyperparameters, of type `M`), the `fitresult` needed for prediction, and the
fit `report` (see `modelreports`). `P` is the tuple of predictor names,
which `applymodels` predicts from. Serialization, as in
`writejls`, goes through MLJ's `save`/`restore`.
