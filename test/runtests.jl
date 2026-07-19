using Aqua
using CausalFrames
using DataFrames
using Dates
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
    include("stream.jl")
    include("join.jl")
    include("segtree.jl")
    include("rolling.jl")
    include("summarize.jl")
    include("summarizers.jl")
end
