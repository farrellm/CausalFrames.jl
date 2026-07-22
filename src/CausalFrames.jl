module CausalFrames

using CSV
using DataFrames
using PrecompileTools: @setup_workload, @compile_workload
using Tables

export Context, CausalFrame, CausalPipeline, load, stream, context, timetype,
    emptyframe, clock, readcsv, filterrows, addcolumns, selectcolumns,
    dropcolumns,
    Summarizer, MonoidSummarizer, GroupSummarizer, SummarizerState,
    Count, Sum, SumPower, Moment, Product,
    DotProduct, Mean, Variance, Std, Covariance, Correlation, Min, Max,
    First, Last, summarize, summarizecycles, addsummarycolumns,
    addrollingcolumns, asofjoin

include("context.jl")
include("frame.jl")
include("chunks.jl")
include("pipeline.jl")
include("operators.jl")
include("summarizers.jl")
include("summarize.jl")
include("join.jl")
include("segtree.jl")
include("rolling.jl")
include("precompile.jl")

end
