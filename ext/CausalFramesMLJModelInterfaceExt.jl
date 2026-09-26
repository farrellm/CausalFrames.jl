# The MLJ backend: the model hooks of src/models.jl for any MLJModelInterface
# model. Fitting and prediction use MLJ's model-level API (`fit`/`predict` over
# the model's `reformat` front-end), not machines, so MLJBase isn't needed
# here; most model implementations need it themselves (for
# `MLJModelInterface.matrix` and friends), which is why users load `using MLJ`.
module CausalFramesMLJModelInterfaceExt

using CausalFrames
using MLJModelInterface: MLJModelInterface as MMI

CausalFrames.ismodel(::MMI.Model) = true

# The report is normalized as MLJBase's `report(mach)` does for a freshly fit
# machine (`MMI.report` over the fit report alone), so `modelreports` agrees
# with it: an empty report is `nothing`, and a model's `report` overload is
# honoured.
function CausalFrames.fitmodel(model::MMI.Model, verbosity::Int, X, y)
    fitresult, _, report = MMI.fit(model, verbosity, MMI.reformat(model, X, y)...)
    return fitresult, MMI.report(model, Dict{Symbol,Any}(:fit => report))
end

# The operation is validated against CausalFrames.PREDICTOPS at construction.
# The probabilistic summaries are stubs here with fallbacks in MLJBase, so an
# unsupported one surfaces as MLJ's own MethodError.
predictop(::Val{:predict}) = MMI.predict
predictop(::Val{:predict_mean}) = MMI.predict_mean
predictop(::Val{:predict_mode}) = MMI.predict_mode
predictop(::Val{:predict_median}) = MMI.predict_median

CausalFrames.predictmodel(model::MMI.Model, fitresult, op::Symbol, X) =
    predictop(Val(op))(model, fitresult, MMI.reformat(model, X)...)

# MLJModelInterface declares save/restore with no methods (MLJBase has the
# identity fallbacks), so they are called only where a method applies.
CausalFrames.savefitresult(model::MMI.Model, fitresult) =
    applicable(MMI.save, model, fitresult) ? MMI.save(model, fitresult) :
    fitresult
CausalFrames.restorefitresult(model::MMI.Model, stored) =
    applicable(MMI.restore, model, stored) ? MMI.restore(model, stored) : stored

end
