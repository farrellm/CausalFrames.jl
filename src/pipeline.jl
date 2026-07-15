"""
    CausalPipeline(run)

A lazy description of how to produce a [`CausalFrame`](@ref): conceptually
a function `Context -> CausalFrame`. Build pipelines from sources
([`emptyframe`](@ref), [`clock`](@ref), [`readcsv`](@ref)) and chain
transforms with `|>`:

```julia
p = readcsv("ticks.csv") |>
    filterrows(r -> r.price > 0) |>
    addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
```

Every operator must be *causal*: its output at time `t` may depend only on
input rows with time `<= t`.
"""
struct CausalPipeline
    run::Function
end

"""
    load(ctx::Context, p::CausalPipeline) -> CausalFrame

Evaluate the pipeline over the time window `ctx`, materializing a frame.
"""
load(ctx::Context, p::CausalPipeline) = p.run(ctx)::CausalFrame
