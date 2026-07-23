# Benchmarks for the hot paths: row-wise transforms, the summarization
# kernels, and the sources. PkgBenchmark-compatible (defines SUITE), and
# runnable directly:
#
#     julia --project=benchmark -e 'using Pkg; Pkg.develop(path=".")'
#     julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using CausalFrames
using CausalFrames.Acausal: futurejoin
using DataFrames

# A deterministic keyed trades table served in chunks, like a real source.
# Duplicate times (4 rows per timestamp) exercise summarizecycles.
function tradesource(n; chunkrows = 100_000, nkeys = 100)
    return CausalPipeline() do ctx
        return (
            DataFrame(time = collect(r) .÷ 4,
                sym = ["s" * string(i % nkeys) for i in r],
                qty = [1.0 + (i % 7) for i in r])
            for r in Iterators.partition(0:(n-1), chunkrows)
        )
    end
end

const N = 1_000_000
const CTX = Context(0, N)
const SRC = tradesource(N)
# A second, independent source for the binary joins, so the right side is not
# a self join (which would need a prefix).
const SRC2 = tradesource(N)

const CSVPATH = joinpath(mktempdir(), "bench.csv")
open(CSVPATH, "w") do io
    println(io, "time,qty")
    for t in 1:200_000
        println(io, "$t,$(1.0 + t % 7)")
    end
end

# A structure-hiding wrapper: delegates the interface to the wrapped
# summarizer but subtypes plain Summarizer, so addrollingcolumns takes its
# re-fold fallback — the baseline the fast paths are measured against.
struct RefoldWrap{S<:CausalFrames.Summarizer} <: CausalFrames.Summarizer
    inner::S
end

CausalFrames.emptyvalue(o::RefoldWrap) = CausalFrames.emptyvalue(o.inner)
CausalFrames.fresh(o::RefoldWrap, intypes::NamedTuple) =
    CausalFrames.fresh(o.inner, intypes)
CausalFrames.dependencies(o::RefoldWrap) =
    map(RefoldWrap, CausalFrames.dependencies(o.inner))

# A smaller source for the rolling benchmarks: the refold baseline is
# O(window) per row, so a million-row input would dominate the suite.
const RN = 100_000
const RCTX = Context(0, RN)
const RSRC = tradesource(RN)

const SUITE = BenchmarkGroup()

SUITE["sources"] = BenchmarkGroup()
SUITE["sources"]["clock"] = @benchmarkable load(CTX, clock(1))
SUITE["sources"]["readcsv"] = @benchmarkable load(Context(0, 300_000),
    readcsv(CSVPATH; types = Dict(:time => Int, :qty => Float64)))

SUITE["rowwise"] = BenchmarkGroup()
SUITE["rowwise"]["filterrows"] =
    @benchmarkable load(CTX, SRC |> filterrows(r -> r.qty > 3.0))
SUITE["rowwise"]["addcolumns"] = @benchmarkable load(CTX,
    SRC |> addcolumns(r -> (; v = r.qty * 2.0, w = r.qty + 1.0)))
SUITE["rowwise"]["selectcolumns"] =
    @benchmarkable load(CTX, SRC |> selectcolumns(:qty))
SUITE["rowwise"]["dropcolumns"] =
    @benchmarkable load(CTX, SRC |> dropcolumns(startswith("s")))
SUITE["rowwise"]["pipeline"] = @benchmarkable load(
    CTX,
    clock(1) |>
    filterrows(r -> r.time % 3 != 0) |> addcolumns(r -> (; x = 0.5 * r.time)),
)

SUITE["summarize"] = BenchmarkGroup()
SUITE["summarize"]["keyless"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty), Min(:qty), Max(:qty)]))
SUITE["summarize"]["keyed"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty)]; key = :sym))
SUITE["summarize"]["cycles"] = @benchmarkable load(CTX,
    SRC |> summarizecycles([Count(), Sum(:qty)]))
SUITE["summarize"]["running"] = @benchmarkable load(CTX,
    SRC |> addsummarycolumns([Sum(:qty), Last(:qty)]))

# intervalize over the same source and summarizers as the summarize group, so
# the two are directly comparable: the difference is the per-interval state
# reset, boundary crossing, and grid emission on top of the same per-row fold.
# The clock carves the window into ~1000 intervals — negligible against the
# million rows folded — of which the data (times 0..N÷4) fills the first
# quarter, so the keyless grid also exercises empty-interval emission.
SUITE["intervalize"] = BenchmarkGroup()
SUITE["intervalize"]["keyless"] = @benchmarkable load(CTX,
    SRC |> intervalize(clock(1000), [Count(), Sum(:qty), Min(:qty), Max(:qty)]))
SUITE["intervalize"]["keyed"] = @benchmarkable load(CTX,
    SRC |> intervalize(clock(1000), [Count(), Sum(:qty)]; key = :sym))
SUITE["intervalize"]["closelast"] = @benchmarkable load(CTX,
    SRC |> intervalize(clock(1000), [Count(), Sum(:qty)]; closelast = true))

# One group per window algorithm: all-group summarizers slide running
# states, all-monoid sets fold from a segment tree, and an unstructured
# summarizer forces the re-fold baseline (see src/rolling.jl).
SUITE["rolling"] = BenchmarkGroup()
SUITE["rolling"]["running"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Sum(:qty), Mean(:qty)]))
SUITE["rolling"]["running-keyed"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Sum(:qty), Mean(:qty)];
        key = :sym))
SUITE["rolling"]["tree"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Min(:qty), Max(:qty)]))
SUITE["rolling"]["refold"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [RefoldWrap(Sum(:qty))]))

# The causal as-of join against its acausal forward mirror, over the same two
# sources. futurejoin's per-key row buffers are the cost the comparison
# exposes, against asofjoin's single-row-per-key store.
SUITE["join"] = BenchmarkGroup()
SUITE["join"]["asof"] =
    @benchmarkable load(CTX, SRC |> asofjoin(SRC2; rightprefix = "r"))
SUITE["join"]["future"] =
    @benchmarkable load(CTX, SRC |> futurejoin(SRC2; rightprefix = "r"))
SUITE["join"]["asof-keyed"] = @benchmarkable load(CTX,
    SRC |> asofjoin(SRC2; key = :sym, rightprefix = "r"))
SUITE["join"]["future-keyed"] = @benchmarkable load(CTX,
    SRC |> futurejoin(SRC2; key = :sym, rightprefix = "r"))

if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    for (path, trial) in BenchmarkTools.leaves(results)
        t = BenchmarkTools.prettytime(time(median(trial)))
        m = BenchmarkTools.prettymemory(memory(trial))
        println(rpad(join(path, "/"), 28), lpad(t, 12), lpad(m, 12),
            lpad(allocs(trial), 12), " allocs")
    end
end
