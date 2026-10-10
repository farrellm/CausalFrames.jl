# Peak live heap of the rolling lead-in cases (`LEADCASES` in benchmarks.jl),
# which BenchmarkTools' allocation totals don't show: a chunk held for later
# costs nothing extra to allocate, only to keep. Runnable directly, after the
# same one-time setup as benchmarks.jl:
#
#     julia -t 1 --project=benchmark benchmark/peakmemory.jl
#
# A probe after the source runs a full GC at every chunk the source yields and
# samples the live heap, so the peak is exact at chunk granularity. The
# pipeline is drained with `scan`, not `load`: a loaded output grows to about
# the lead-in's size by the end of the run and would mask it.

include("benchmarks.jl")

function peaklive(ctx::Context, build)
    peak = Ref(0)
    probe(q) = CausalPipeline() do c
        CausalFrames.chunkmap(q.run(c)) do chunk
            GC.gc()
            peak[] = max(peak[], Base.gc_live_bytes())
            return chunk
        end
    end
    p = build(probe(LEADSRC))
    scan(ctx, p)                     # compile outside the measurement
    GC.gc()
    base = Base.gc_live_bytes()
    peak[] = base
    scan(ctx, p)
    return peak[] - base
end

println(rpad("case", 30), lpad("shared", 12), lpad("two runs", 12))
for (name, f) in LEADCASES
    mib(sh) = string(round(peaklive(LEADCTX, p -> f(p, sh)) / 2^20;
        digits = 1), " MiB")
    println(rpad(name, 30), lpad(mib(true), 12), lpad(mib(false), 12))
end
