# The derived states are fieldless (their value comes from the dependencies'
# at emission), so folding, combining and downdating are no-ops. The window
# transforms give them no tier (tiers.jl) except in a set made only of
# fieldless states, which these methods serve.
fresh(st::DerivedState) = st
@inline update!(::DerivedState, row) = nothing
combine!(::DerivedState, ::DerivedState, ::DerivedState) = nothing
@inline downdate!(::DerivedState, row) = nothing
# Fieldless, so already zero; defined so `freshall!` needs no call to `fresh`.
@inline fresh!(st::DerivedState) = st
