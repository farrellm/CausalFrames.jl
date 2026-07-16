module CausalFrames

using CSV
using DataFrames
using Tables

export Context, CausalFrame, CausalPipeline, load, context, timetype,
    emptyframe, clock, readcsv, filterrows, addcolumns,
    Summarizer, Count, Sum, summarize, summarizecycles, addsummarycolumns

include("context.jl")
include("frame.jl")
include("pipeline.jl")
include("operators.jl")
include("summarize.jl")

end
