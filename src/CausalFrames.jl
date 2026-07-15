module CausalFrames

using CSV
using DataFrames
using Tables

export Context, CausalFrame, CausalPipeline, load, context, timetype,
    emptyframe, clock, readcsv, filterrows, addcolumns

include("context.jl")
include("frame.jl")
include("pipeline.jl")
include("operators.jl")

end
