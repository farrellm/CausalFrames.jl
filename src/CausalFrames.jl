module CausalFrames

using CSV
using DataFrames
using PrecompileTools: @setup_workload, @compile_workload
using Tables

export Context, CausalFrame, CausalPipeline, load, stream, context, timetype,
    emptyframe, clock, readcsv, filterrows, addcolumns,
    Summarizer, SummarizerState, Count, Sum, SumPower, Min, Max, First, Last,
    summarize, summarizecycles, addsummarycolumns

include("context.jl")
include("frame.jl")
include("chunks.jl")
include("pipeline.jl")
include("operators.jl")
include("summarizers.jl")
include("summarize.jl")
include("precompile.jl")

end
