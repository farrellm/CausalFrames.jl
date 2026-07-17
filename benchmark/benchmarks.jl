# Benchmarks for the hot paths: row-wise transforms, the summarization
# kernels, and the sources. PkgBenchmark-compatible (defines SUITE), and
# runnable directly:
#
#     julia --project=benchmark -e 'using Pkg; Pkg.develop(path=".")'
#     julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using CausalFrames
using DataFrames

# A deterministic keyed trades table served in chunks, like a real source.
# Duplicate times (4 rows per timestamp) exercise summarizecycles.
function tradesource(n; chunkrows = 100_000, nkeys = 100)
    return CausalPipeline() do ctx
        return (DataFrame(time = collect(r) .÷ 4,
                          sym = ["s" * string(i % nkeys) for i in r],
                          qty = [1.0 + (i % 7) for i in r])
                for r in Iterators.partition(0:(n - 1), chunkrows))
    end
end

const N = 1_000_000
const CTX = Context(0, N)
const SRC = tradesource(N)

const CSVPATH = joinpath(mktempdir(), "bench.csv")
open(CSVPATH, "w") do io
    println(io, "time,qty")
    for t in 1:200_000
        println(io, "$t,$(1.0 + t % 7)")
    end
end

const SUITE = BenchmarkGroup()

SUITE["sources"] = BenchmarkGroup()
SUITE["sources"]["clock"] = @benchmarkable load(CTX, clock(1))
SUITE["sources"]["readcsv"] =
    @benchmarkable load(Context(0, 300_000), readcsv(CSVPATH))

SUITE["rowwise"] = BenchmarkGroup()
SUITE["rowwise"]["filterrows"] =
    @benchmarkable load(CTX, SRC |> filterrows(r -> r.qty > 3.0))
SUITE["rowwise"]["addcolumns"] = @benchmarkable load(CTX,
    SRC |> addcolumns(r -> (; v = r.qty * 2.0, w = r.qty + 1.0)))
SUITE["rowwise"]["pipeline"] = @benchmarkable load(CTX, clock(1) |>
    filterrows(r -> r.time % 3 != 0) |> addcolumns(r -> (; x = 0.5 * r.time)))

SUITE["summarize"] = BenchmarkGroup()
SUITE["summarize"]["keyless"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty), Min(:qty), Max(:qty)]))
SUITE["summarize"]["keyed"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty)]; key = :sym))
SUITE["summarize"]["cycles"] = @benchmarkable load(CTX,
    SRC |> summarizecycles([Count(), Sum(:qty)]))
SUITE["summarize"]["running"] = @benchmarkable load(CTX,
    SRC |> addsummarycolumns([Sum(:qty), Last(:qty)]))

if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    for (path, trial) in leaves(results)
        t = BenchmarkTools.prettytime(time(median(trial)))
        m = BenchmarkTools.prettymemory(memory(trial))
        println(rpad(join(path, "/"), 28), lpad(t, 12), lpad(m, 12),
                lpad(allocs(trial), 12), " allocs")
    end
end
