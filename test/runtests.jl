using Aqua
using CausalFrames
using DataFrames
using Dates
using DuckDB
using Parquet2
using Tables
using Test

include("fixtures.jl")

@testset "CausalFrames.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(CausalFrames)
    end

    include("context.jl")
    include("frame.jl")
    include("operators.jl")
    include("parquet.jl")
    include("stream.jl")
    include("join.jl")
    include("segtree.jl")
    include("rolling.jl")
    include("summarize.jl")
    include("summarizers.jl")

    # JET can lag pre-release Julia; the checks are the same on every
    # released version, so skipping them there loses nothing.
    if isempty(VERSION.prerelease)
        include("jet.jl")
    end
end
