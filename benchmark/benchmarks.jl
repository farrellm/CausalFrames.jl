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

# A deterministic keyed trades table, built **once** and served in chunks like
# a real source. Duplicate times (4 rows per timestamp) exercise
# summarizecycles, and the symbols stay `String`s, as from a real source, so
# the keyed paths hash String-carrying keys.
#
# Generation is hoisted to load time so only the pipeline is timed: building a
# million `"s" * string(i)` symbols costs ~3 allocations per row, which would
# bury the operators' own cost.
function tradechunks(n; chunkrows = 100_000, nkeys = 100)
    syms = ["s" * string(k) for k in 0:(nkeys-1)]   # interned once
    return [
        DataFrame(time = collect(r) .÷ 4,
            sym = [@inbounds(syms[i%nkeys+1]) for i in r],
            qty = [1.0 + (i % 7) for i in r])
        for r in Iterators.partition(0:(n-1), chunkrows)
    ]
end

# Consumers own their chunks and may mutate a chunk's column *index*, so each
# run gets a private index over the same column vectors, O(ncols) per chunk.
# Column vectors are never mutated in place (see DESIGN.md, "CSV output"), so
# sharing them is sound.
tradesource(chunks) =
    CausalPipeline(ctx -> (DataFrame(c; copycols = false) for c in chunks))

const N = 1_000_000
const CTX = Context(0, N)
const CHUNKS = tradechunks(N)
const SRC = tradesource(CHUNKS)
# A second, independent source for the binary joins, so the right side is not
# a self join (which would need a prefix).
const SRC2 = tradesource(tradechunks(N))

const BENCHDIR = mktempdir()
const CSVPATH = joinpath(BENCHDIR, "bench.csv")
const SINKPATH = joinpath(BENCHDIR, "sink.csv")
const CSVTYPES = Dict(:time => Int, :qty => Float64)
open(CSVPATH, "w") do io
    println(io, "time,qty")
    for t in 1:200_000
        println(io, "$t,$(1.0 + t % 7)")
    end
end

# A structure-hiding wrapper: delegates to the wrapped summarizer but subtypes
# plain Summarizer, so the window transforms re-fold it, the baseline for the
# running and tree tiers.
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
const RSRC = tradesource(tradechunks(RN))

const SUITE = BenchmarkGroup()

# The pass-through sink: `scan` never materializes a frame, so against the
# drain floor this measures the background writer hand-off and CSV formatting.
# Over the smaller source, since the cost is dominated by disk I/O. The source
# ignores the context (it hands out fixed chunks), so the window must be the
# one that actually contains its rows.
SUITE["sinks"] = BenchmarkGroup()
SUITE["sinks"]["writecsv"] =
    @benchmarkable scan(RCTX, RSRC |> writecsv(SINKPATH))

SUITE["sources"] = BenchmarkGroup()
SUITE["sources"]["clock"] = @benchmarkable load(CTX, clock(1))
SUITE["sources"]["readcsv"] = @benchmarkable load(Context(0, 300_000),
    readcsv(CSVPATH; types = CSVTYPES))
# The floor every SRC-based entry below sits on: chunk hand-off, the per-chunk
# load guards, and frame assembly, with no operator in the chain. Subtract it
# to read an operator's own cost.
SUITE["sources"]["drain"] = @benchmarkable scan(CTX, SRC)
SUITE["sources"]["drain-load"] = @benchmarkable load(CTX, SRC)

# The in-memory source over one million-row table, as a DataFrame (resolved
# once, when readtable is called), as a NamedTuple of the same vectors (the
# generic path, resolved per run), and as a loaded frame (whole-window chunks
# shared, not copied) — each over the whole window and over a 1% window. The
# pipelines are built outside the timed expression, as a user reusing one would.
const TABLE = reduce(vcat, CHUNKS)
const TABLENT = NamedTuple{Tuple(propertynames(TABLE))}(Tuple(eachcol(TABLE)))
const TABLEFRAME = load(CTX, SRC)
const NARROW = Context(100_000, 102_500)
for (name, p) in (("dataframe", readtable(TABLE)), ("columntable", readtable(TABLENT)),
    ("frame", readtable(TABLEFRAME)))
    SUITE["sources"]["readtable-$name"] = @benchmarkable load(CTX, $p)
    SUITE["sources"]["readtable-$name-narrow"] = @benchmarkable load(NARROW, $p)
end

SUITE["rowwise"] = BenchmarkGroup()
SUITE["rowwise"]["filterrows"] =
    @benchmarkable load(CTX, SRC |> filterrows(r -> r.qty > 3.0))
SUITE["rowwise"]["addcolumns"] = @benchmarkable load(CTX,
    SRC |> addcolumns(r -> (; v = r.qty * 2.0, w = r.qty + 1.0)))
SUITE["rowwise"]["selectcolumns"] =
    @benchmarkable load(CTX, SRC |> selectcolumns(:qty))
SUITE["rowwise"]["dropcolumns"] =
    @benchmarkable load(CTX, SRC |> dropcolumns(startswith("s")))
SUITE["rowwise"]["reordercolumns"] =
    @benchmarkable load(CTX, SRC |> reordercolumns(:qty))
SUITE["rowwise"]["pipeline"] = @benchmarkable load(
    CTX,
    clock(1) |>
    filterrows(r -> r.time % 3 != 0) |> addcolumns(r -> (; x = 0.5 * r.time)),
)
# Two entries, because they measure different things. Over SRC the cost is the
# one partial-chunk slice: SRC hands out pre-built chunks, so a head that
# failed to stop early would look the same. Over readcsv it is the early exit
# itself. `chunkbytes` is set because the 4 MiB default swallows this 200k-row
# file whole; at 64 KiB the file is ~45 chunks and head(1000) should read one,
# far under `head-drain`, the same read without the early stop.
SUITE["rowwise"]["head"] = @benchmarkable load(CTX, SRC |> head(1000))
SUITE["rowwise"]["head-early-exit"] = @benchmarkable load(Context(0, 300_000),
    readcsv(CSVPATH; types = CSVTYPES, chunkbytes = 65_536) |> head(1000))
SUITE["rowwise"]["head-drain"] = @benchmarkable load(Context(0, 300_000),
    readcsv(CSVPATH; types = CSVTYPES, chunkbytes = 65_536))
# settime's per-row work: the forward check over two concrete vectors, plus one
# maptime pass in the function form.
SUITE["rowwise"]["settime-column"] = @benchmarkable load(CTX,
    SRC |> addcolumns(r -> (; t2 = r.time + 1)) |> settime(:t2))
SUITE["rowwise"]["settime-function"] =
    @benchmarkable load(CTX, SRC |> settime(r -> r.time + 1))
# sortcycles over SRC's four-row cycles, one hold-back per chunk. `qty` rises
# within most cycles, so the reversed column and the negating function reorder
# nearly every one of them; `sorted` keys on `:time`, which ties throughout, so
# it reads the issorted check and the prefix copy alone.
SUITE["rowwise"]["sortcycles-column"] =
    @benchmarkable load(CTX, SRC |> sortcycles(:qty; rev = true))
SUITE["rowwise"]["sortcycles-function"] =
    @benchmarkable load(CTX, SRC |> sortcycles(r -> -r.qty))
SUITE["rowwise"]["sortcycles-sorted"] =
    @benchmarkable load(CTX, SRC |> sortcycles(:time))

# One hash and one inline row store per row on the keyed path, directly
# comparable to summarize/keyed over the same source and keys; keyless does
# nothing per row and should read as the drain floor. SRC's `sym` column is a
# String, which is the non-isbits V the store's design exists for.
SUITE["lastrow"] = BenchmarkGroup()
SUITE["lastrow"]["keyless"] = @benchmarkable load(CTX, SRC |> lastrow())
SUITE["lastrow"]["keyed"] = @benchmarkable load(CTX, SRC |> lastrow(; key = :sym))

SUITE["summarize"] = BenchmarkGroup()
SUITE["summarize"]["keyless"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty), Min(:qty), Max(:qty)]))
SUITE["summarize"]["keyed"] = @benchmarkable load(CTX,
    SRC |> summarize([Count(), Sum(:qty)]; key = :sym))
# The squared power sum, which nothing else in this suite reaches: every
# Variance, Std, Covariance, Correlation and LinearRegression folds one, so its
# per-row cost is worth watching (see notes/sumpower-terms.md).
SUITE["summarize"]["powersum"] = @benchmarkable load(CTX,
    SRC |> summarize([SumPower(:qty, 2), Variance(:qty)]))
SUITE["summarize"]["cycles"] = @benchmarkable load(CTX,
    SRC |> summarizecycles([Count(), Sum(:qty)]))
# The keyed cycle fold closes a group table per timestamp — 250k cycles over
# this source — so it is the path where per-cycle state churn shows up.
SUITE["summarize"]["cycles-keyed"] = @benchmarkable load(CTX,
    SRC |> summarizecycles([Count(), Sum(:qty)]; key = :sym))
SUITE["summarize"]["running"] = @benchmarkable load(CTX,
    SRC |> addsummarycolumns([Sum(:qty), Last(:qty)]))
SUITE["summarize"]["countdistinct"] = @benchmarkable load(CTX,
    SRC |> summarize(CountDistinct(:qty); key = :sym))

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
# Every symbol declared, so each of the ~1000 intervals emits a row per symbol,
# empty ones included: 100,000 rows where the sparse keyed entry emits ~25,000.
# (There is no dense summarizecycles entry: 250k cycles × 100 symbols is 25M
# output rows, which would time DataFrame construction rather than the fold.)
const SYMS = ["s" * string(k) for k in 0:99]
SUITE["intervalize"]["dense"] = @benchmarkable load(CTX,
    SRC |> intervalize(clock(1000), [Count(), Sum(:qty)]; key = :sym,
        keyset = SYMS))

# The same ~1000 ticks, each summarizing a trailing 5000 time units (about
# 20,000 rows, five tick spacings, so windows overlap five-fold). Entries per
# window tier: the running tier slides per-key states once per row (Min/Max
# through their windowed deques in "tracking"); the tree tier (Product is only
# a monoid) appends rows to a segment tree and recombines them at the tick; the
# re-fold baseline folds every window at its tick, paying the overlap. "mixed"
# is an OHLC-style set spanning the running tier and dependents, at a look-back
# shorter than the tick spacing and at 5000.
SUITE["windows"] = BenchmarkGroup()
SUITE["windows"]["running"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Count(), Sum(:qty), Mean(:qty)]))
SUITE["windows"]["running-keyed"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Count(), Sum(:qty)];
        key = :sym))
# The same with every symbol declared: a row per symbol per tick, looked up in
# declared order instead of sorted from the live groups.
SUITE["windows"]["running-dense"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Count(), Sum(:qty)];
        key = :sym, keyset = SYMS))
SUITE["windows"]["tracking"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Min(:qty), Max(:qty)]))
SUITE["windows"]["tracking-keyed"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Min(:qty), Max(:qty)];
        key = :sym))
SUITE["windows"]["tree"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000, [Product(:qty)]))
const MIXED = [First(:qty), Max(:qty), Min(:qty), Last(:qty), Mean(:qty),
    Std(:qty)]
for L in (20, 5000)
    SUITE["windows"]["mixed-$L"] = @benchmarkable load(CTX,
        SRC |> summarizewindows(clock(1000), $L, MIXED))
    SUITE["windows"]["mixed-keyed-$L"] = @benchmarkable load(CTX,
        SRC |> summarizewindows(clock(1000), $L, MIXED; key = :sym))
end
SUITE["windows"]["refold"] = @benchmarkable load(CTX,
    SRC |> summarizewindows(clock(1000), 5000,
        [RefoldWrap(Min(:qty)), RefoldWrap(Max(:qty))]))

# Entries per window tier, as for the windows group (see src/tiers.jl), plus
# the mixed set at windows of about 4, 24 and 1000 rows (keyless); the small
# ones are where a running+tree split could lose to a tree alone.
SUITE["rolling"] = BenchmarkGroup()
SUITE["rolling"]["running"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Sum(:qty), Mean(:qty)]))
SUITE["rolling"]["running-keyed"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Sum(:qty), Mean(:qty)];
        key = :sym))
SUITE["rolling"]["tracking"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Min(:qty), Max(:qty)]))
SUITE["rolling"]["countdistinct"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [CountDistinct(:qty)]))
SUITE["rolling"]["quantile"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25),
        [Quantile(:qty, [0.25, 0.5, 0.75]), PercentRank(:qty)]))
SUITE["rolling"]["tree"] = @benchmarkable load(RCTX,
    RSRC |> addrollingcolumns((; w25 = 25), [Product(:qty)]))
for (nm, w) in (("w0", 0), ("w5", 5), ("w250", 250))
    SUITE["rolling"]["mixed-$nm"] = @benchmarkable load(RCTX,
        RSRC |> addrollingcolumns((; a = $w), MIXED))
    SUITE["rolling"]["mixed-keyed-$nm"] = @benchmarkable load(RCTX,
        RSRC |> addrollingcolumns((; a = $w), MIXED; key = :sym))
end
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

# The timeless lookup, against the keyed as-of join above: a static index, one
# Int per row and a gather per column. "lookup-drop" lists half the keys, so
# every chunk takes the row-dropping slice.
const DIM = DataFrame(sym = ["s" * string(k) for k in 0:99],
    lot = [Float64(k) for k in 0:99],
    sector = ["sector" * string(k % 10) for k in 0:99])
const HALFDIM = DIM[1:2:end, :]
SUITE["join"]["lookup-keyed"] =
    @benchmarkable load(CTX, SRC |> lookupjoin(DIM; key = :sym))
SUITE["join"]["lookup-drop"] = @benchmarkable load(CTX,
    SRC |> lookupjoin(HALFDIM; key = :sym, unmatched = :drop))

# The n-ary merge over the same two sources. "interleaved" is the worst case:
# the streams share every timestamp, so blocks are cycle-sized and rows are
# copied. "shifted" moves one source a half-window later so long runs of whole
# chunks pass through untouched. "union" merges sources with different
# columns, the missing-filling path.
SUITE["merge"] = BenchmarkGroup()
SUITE["merge"]["interleaved"] = @benchmarkable load(CTX, merge(SRC, SRC2))
SUITE["merge"]["shifted"] = @benchmarkable load(CTX,
    merge(SRC, SRC2 |> lag(N ÷ 8)))
SUITE["merge"]["union"] = @benchmarkable load(CTX,
    merge(SRC |> dropcolumns(:sym), SRC2 |> dropcolumns(:qty)))

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
