# The summarizer interface and the concrete summarizers, one file per family.
# Include order matters only for types: `Compensated` precedes the states
# holding it, and `DerivedState` follows every derived state.
include("summarizers/interface.jl")
include("summarizers/compensated.jl")
include("summarizers/sums.jl")
include("summarizers/comoment.jl")
include("summarizers/agesum.jl")
include("summarizers/moments.jl")
include("summarizers/regression.jl")
include("summarizers/orderstats.jl")
include("summarizers/windowvalues.jl")
include("summarizers/derived.jl")
include("summarizers/extremes.jl")
include("summarizers/fitmodel.jl")
include("summarizers/inputcolumns.jl")
