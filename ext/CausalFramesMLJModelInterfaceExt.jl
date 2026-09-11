# The MLJ backend: the hook methods src/models.jl's FitModel, applymodels and
# the FittedModel serializer call, for any MLJModelInterface model. Fitting and
# prediction go through MLJ's model-level API (`fit`/`predict` with the model's
# data front-end, `reformat`) rather than machines, so MLJBase is not needed
# here — though most model implementations need it themselves, for
# `MLJModelInterface.matrix` and friends, which is why users load `using MLJ`.
module CausalFramesMLJModelInterfaceExt

using CausalFrames
using MLJModelInterface: MLJModelInterface as MMI

CausalFrames.ismodel(::MMI.Model) = true

# The report is normalized exactly as MLJBase's `report(mach)` normalizes a
# freshly fit machine's — `MMI.report` over the fit report alone — so
# `modelreports` and `report(mach)` agree: an empty report becomes `nothing`,
# and a model overloading `report` is honoured.
function CausalFrames.fitmodel(model::MMI.Model, verbosity::Int, X, y)
    fitresult, _, report = MMI.fit(model, verbosity, MMI.reformat(model, X, y)...)
    return fitresult, MMI.report(model, Dict{Symbol,Any}(:fit => report))
end

# The operation is validated against CausalFrames.PREDICTOPS at construction;
# the probabilistic summaries are stubs in MLJModelInterface itself, with
# fallbacks in MLJBase, so an unsupported one surfaces as MLJ's own MethodError.
predictop(::Val{:predict}) = MMI.predict
predictop(::Val{:predict_mean}) = MMI.predict_mean
predictop(::Val{:predict_mode}) = MMI.predict_mode
predictop(::Val{:predict_median}) = MMI.predict_median

CausalFrames.predictmodel(model::MMI.Model, fitresult, op::Symbol, X) =
    predictop(Val(op))(model, fitresult, MMI.reformat(model, X)...)

# MLJModelInterface declares save/restore with no methods at all — the identity
# fallbacks live in MLJBase — so they are called only where a method applies:
# a model implementing them, or any model once MLJBase is loaded.
CausalFrames.savefitresult(model::MMI.Model, fitresult) =
    applicable(MMI.save, model, fitresult) ? MMI.save(model, fitresult) :
    fitresult
CausalFrames.restorefitresult(model::MMI.Model, stored) =
    applicable(MMI.restore, model, stored) ? MMI.restore(model, stored) : stored

end
