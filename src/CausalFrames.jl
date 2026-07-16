module CausalFrames

using CSV
using DataFrames
using Tables

export Context, CausalFrame, CausalPipeline, load, stream, context, timetype,
    emptyframe, clock, readcsv, filterrows, addcolumns,
    Summarizer, Count, Sum, SumPower, Min, Max, First, Last,
    summarize, summarizecycles, addsummarycolumns

include("context.jl")
include("frame.jl")
include("chunks.jl")
include("pipeline.jl")
include("operators.jl")
include("summarize.jl")

end
