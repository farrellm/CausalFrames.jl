using Aqua
using CausalFrames
using DataFrames
using Dates
using DuckDB
# Imported, not used: MLJModelInterface re-exports the scientific types, and
# its `Count` would clash with the summarizer (as MLJ's does for users).
import MLJModelInterface
using Parquet2
using Serialization
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
    include("jls.jl")
    include("stream.jl")
    include("merge.jl")
    include("join.jl")
    include("lastrow.jl")
    include("fill.jl")
    include("acausal.jl")
    include("segtree.jl")
    include("rolling.jl")
    include("intervalize.jl")
    include("windows.jl")
    include("summarize.jl")
    include("summarizers.jl")
    include("models.jl")

    # JET can lag pre-release Julia; the checks are the same on every
    # released version, so skipping them there loses nothing.
    if isempty(VERSION.prerelease)
        include("jet.jl")
    end
end
