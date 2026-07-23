module CausalFrames

using CSV
using DataFrames
using PrecompileTools: @setup_workload, @compile_workload
using Tables

export Context, CausalFrame, CausalPipeline, load, stream, scan, context,
    timetype,
    emptyframe, clock, readcsv, writecsv, readparquet, writeparquet,
    filterrows, addcolumns,
    selectcolumns, dropcolumns,
    Summarizer, MonoidSummarizer, GroupSummarizer, SummarizerState,
    Count, Sum, SumPower, Moment, Product,
    DotProduct, Mean, Variance, Std, Covariance, Correlation, Min, Max,
    First, Last, summarize, summarizecycles, addsummarycolumns,
    addrollingcolumns, asofjoin, intervalize

include("context.jl")
include("frame.jl")
include("chunks.jl")
include("pipeline.jl")
include("operators.jl")
include("parquet.jl")
include("summarizers.jl")
include("summarize.jl")
include("join.jl")
include("segtree.jl")
include("rolling.jl")
include("intervalize.jl")
include("acausal.jl")
include("precompile.jl")

end
