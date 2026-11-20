using CausalFrames
using DataFrames
using Dates
using Serialization
import Statistics
using Tables
using Test

include("fixtures.jl")

# Balanced test groups so CI can shard across parallel runners via `test_args`
# while a plain `Pkg.test()` (empty ARGS) runs every file. Optional test
# dependencies (Aqua, DuckDB, Parquet2, MLJModelInterface, JET) load only for
# the `core` group that exercises them; `models.jl` precedes `jet.jl` because
# `jet.jl` reuses its `ToyOLS` fixture.
const TEST_GROUPS = (
    "core" => [
        "context.jl",
        "frame.jl",
        "jls.jl",
        "table.jl",
        "stream.jl",
        "merge.jl",
        "join.jl",
        "lookupjoin.jl",
        "lastrow.jl",
        "sortcycles.jl",
        "fill.jl",
        "acausal.jl",
        "parquet.jl",
        "models.jl",
    ],
    "rolling" => [
        "rolling.jl",
    ],
    "windows" => [
        "windows.jl",
        "barwindow.jl",
        "monoidqueue.jl",
        "intervalize.jl",
        "summarize.jl",
        "sharing.jl",
    ],
    "summarizers" => [
        "summarizers.jl",
        "rowterms.jl",
        "keys.jl",
        "operators.jl",
    ],
)

const GROUP_MAP = Dict(TEST_GROUPS)
const SELECTED_GROUPS = isempty(ARGS) ? collect(first.(TEST_GROUPS)) : ARGS
for g in SELECTED_GROUPS
    haskey(GROUP_MAP, g) || throw(
        ArgumentError(
            "unknown test group $(repr(g)); expected one of $(first.(TEST_GROUPS))",
        ),
    )
end

if "core" in SELECTED_GROUPS
    using Aqua
    using DuckDB
    # Imported, not used: MLJModelInterface re-exports the scientific types, and
    # its `Count` would clash with the summarizer (as MLJ's does for users).
    import MLJModelInterface
    using Parquet2
end

@testset "CausalFrames.jl" begin
    if "core" in SELECTED_GROUPS
        @testset "Aqua" begin
            Aqua.test_all(CausalFrames)
        end
    end

    for g in SELECTED_GROUPS
        for f in GROUP_MAP[g]
            include(f)
        end
    end

    # JET can lag pre-release Julia; the checks are the same on every
    # released version, so skipping them there loses nothing.
    if "core" in SELECTED_GROUPS && isempty(VERSION.prerelease)
        include("jet.jl")
    end
end
