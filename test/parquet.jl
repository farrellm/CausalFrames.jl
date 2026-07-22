# The parquet operators, with both backends loaded (runtests.jl brings in
# DuckDB and Parquet2, which activates the extensions).

function writeparquetfile(path, df; kwargs...)
    Parquet2.writefile(path, df; compute_statistics = ["time"], kwargs...)
    return path
end

@testset "readparquet" begin
    dir = mktempdir()
    path = writeparquetfile(joinpath(dir, "ticks.parquet"),
        DataFrame(time = [1, 3, 5, 7], bid = [10.0, 10.5, 9.0, 9.5],
            ask = [11.0, 11.5, 10.0, 10.5]))

    frame = load(Context(0, 100), readparquet(path))
    @test nrow(frame) == 4
    @test names(frame) == ["time", "bid", "ask"]
    @test DataFrame(frame).bid == [10.0, 10.5, 9.0, 9.5]

    # clipped to [start, stop)
    @test DataFrame(load(Context(3, 7), readparquet(path))).time == [3, 5]

    # time column converted to the context's time type
    @test eltype(DataFrame(load(Context(0.0, 100.0), readparquet(path))).time) ==
          Float64

    # types come from the file, not from the caller
    df = DataFrame(load(Context(0, 100), readparquet(path)))
    @test eltype(df.time) == Int
    @test eltype(df.ask) == Float64

    # DateTime times
    stamps = DateTime(2026, 1, 1) .+ Hour.(0:9)
    dtpath = writeparquetfile(joinpath(dir, "dt.parquet"),
        DataFrame(time = stamps, x = 1:10))
    frame = load(Context(DateTime(2026, 1, 1, 2), DateTime(2026, 1, 1, 5)),
        readparquet(dtpath))
    @test DataFrame(frame).time == stamps[3:5]

    unsorted = writeparquetfile(joinpath(dir, "unsorted.parquet"),
        DataFrame(time = [3, 1], x = [1, 2]))
    @test_throws ArgumentError load(Context(0, 100), readparquet(unsorted))

    notime = writeparquetfile(joinpath(dir, "notime.parquet"),
        DataFrame(t = [1, 2], x = [1, 2]))
    @test_throws ArgumentError load(Context(0, 100), readparquet(notime))

    # a textual time column needs a `time` function
    textual = writeparquetfile(joinpath(dir, "textual.parquet"),
        DataFrame(time = ["1", "2"], x = [1, 2]))
    @test_throws ArgumentError load(Context(0, 100), readparquet(textual))
    frame = load(Context(0, 100),
        readparquet(textual; time = row -> parse(Int, row.time)))
    @test DataFrame(frame).time == [1, 2]

    # a column with nulls arrives as Union{Missing,T} and folds
    nulls = writeparquetfile(joinpath(dir, "nulls.parquet"),
        DataFrame(time = 1:4, x = [1.0, missing, 3.0, missing]))
    df = DataFrame(load(Context(0, 100), readparquet(nulls)))
    @test eltype(df.x) == Union{Missing,Float64}
    # a missing-permitting column folds like any other: the missings poison the
    # value but the accumulator stays typed
    summed = DataFrame(
        load(Context(0, 100),
            readparquet(nulls) |> summarize([Count(), Sum(:x)])),
    )
    @test summed.count == [4]
    @test ismissing(only(summed.x_sum))
    @test eltype(summed.x_sum) == Union{Missing,Float64}
end

@testset "readparquet time selection and rename" begin
    dir = mktempdir()
    path = writeparquetfile(joinpath(dir, "ts.parquet"),
        DataFrame(ts = [1, 2, 3], x = [10, 20, 30]))

    frame = load(Context(0, 100), readparquet(path; time = :ts))
    df = DataFrame(frame)
    @test names(df) == ["time", "x"]
    @test df.time == [1, 2, 3]

    @test_throws ArgumentError load(Context(0, 100),
        readparquet(path; time = :nope))

    # time from a function, over the file's own columns
    frame = load(Context(0, 100), readparquet(path; time = row -> row.ts * 2))
    df = DataFrame(frame)
    @test df.time == [2, 4, 6]
    @test df.ts == [1, 2, 3]

    # rename (map), applied before the time column is resolved
    frame = load(Context(0, 100),
        readparquet(path; rename = Dict("ts" => "time")))
    @test DataFrame(frame).time == [1, 2, 3]

    # rename (function)
    frame = load(Context(0, 100),
        readparquet(path; rename = n -> uppercase(String(n)), time = :TS))
    df = DataFrame(frame)
    @test names(df) == ["time", "X"]
    @test df.time == [1, 2, 3]
end

@testset "readparquet window pushdown" begin
    dir = mktempdir()
    times = repeat(1:100, inner = 5)
    frames = DataFrame(time = times, x = float.(1:length(times)))

    # a file with statistics, one without: the pushdown must be invisible
    withstats = writeparquetfile(joinpath(dir, "stats.parquet"), frames)
    without = joinpath(dir, "nostats.parquet")
    Parquet2.writefile(without, frames; compute_statistics = String[])

    ctx = Context(40, 60)
    pushed = DataFrame(load(ctx, readparquet(withstats)))
    unpushed = DataFrame(load(ctx, readparquet(without)))
    # the same window read without any pushdown at all (an opaque time function)
    scanned = DataFrame(load(ctx, readparquet(withstats; time = row -> row.time)))
    @test pushed == unpushed
    @test pushed == scanned
    @test extrema(pushed.time) == (40, 59)

    # a multi-chunk file streams in several chunks, and the window narrows them
    wide = writeparquetfile(joinpath(dir, "wide.parquet"),
        DataFrame(time = 1:20_000, x = 1:20_000))
    allchunks = length(collect(stream(Context(0, 20_000), readparquet(wide))))
    fewchunks = length(collect(stream(Context(0, 100), readparquet(wide))))
    @test allchunks > 1
    @test fewchunks < allchunks
end

@testset "readparquet backends agree" begin
    dir = mktempdir()
    rows = DataFrame(time = 1:100, ints = 1:100, floats = float.(1:100),
        strs = string.(1:100),
        opt = [iseven(i) ? missing : Float64(i) for i in 1:100])

    # ten row groups, so the fallback has something to skip
    function inrowgroups(path, df; stats = ["time"])
        open(path, "w") do io
            fw = Parquet2.FileWriter(io, path; compute_statistics = stats)
            for k in 0:9
                Parquet2.writetable!(fw, df[(k*10+1):(k*10+10), :])
            end
            Parquet2.finalize!(fw)
        end
        return path
    end
    path = inrowgroups(joinpath(dir, "rg.parquet"), rows)

    # every window must read the same rows through either backend (isequal, not
    # ==, because `opt` carries missings)
    for ctx in (Context(0, 200), Context(1, 2), Context(25, 76), Context(95, 200),
        Context(200, 300), Context(0.0, 50.5))
        d = DataFrame(load(ctx, readparquet(path; backend = :duckdb)))
        p = DataFrame(load(ctx, readparquet(path; backend = :parquet2)))
        @test isequal(d, p)
        # :auto prefers DuckDB
        @test isequal(DataFrame(load(ctx, readparquet(path))), d)
    end

    # as must every way of naming the time column
    ctx = Context(25, 76)
    variants = (
        (; time = row -> row.time * 2),
        (; rename = Dict("time" => "t"), time = :t),
        (; rename = n -> uppercase(String(n)), time = :TIME),
    )
    for kw in variants
        d = DataFrame(load(ctx, readparquet(path; backend = :duckdb, kw...)))
        p = DataFrame(load(ctx, readparquet(path; backend = :parquet2, kw...)))
        @test isequal(d, p)
    end

    # a DateTime time column, named by `time`, through both backends
    stamps = DateTime(2026, 1, 1) .+ Hour.(0:99)
    dtpath = inrowgroups(joinpath(dir, "dt.parquet"),
        DataFrame(ts = stamps, x = 1:100); stats = ["ts"])
    dtctx = Context(DateTime(2026, 1, 1, 20), DateTime(2026, 1, 2, 4))
    d = DataFrame(load(dtctx, readparquet(dtpath; backend = :duckdb, time = :ts)))
    p = DataFrame(load(dtctx, readparquet(dtpath; backend = :parquet2, time = :ts)))
    @test d == p
    @test nrow(d) == 8
    # the window falls inside one row group, so the fallback yields one chunk
    @test length(
        collect(stream(dtctx,
            readparquet(dtpath; backend = :parquet2, time = :ts))),
    ) == 1

    # missing-permitting columns survive both readers
    @test eltype(
        DataFrame(load(Context(0, 200),
            readparquet(path; backend = :parquet2))).opt,
    ) == Union{Missing,Float64}

    # the fallback's row-group skipping is invisible in the rows it yields: a
    # file without statistics reads exactly the same window, only more of the
    # file to get there
    plain = inrowgroups(joinpath(dir, "plain.parquet"), rows; stats = String[])
    @test isequal(DataFrame(load(ctx, readparquet(plain; backend = :parquet2))),
        DataFrame(load(ctx, readparquet(path; backend = :parquet2))))
    # groups that clip to nothing yield no chunk either way, so chunk counts
    # cannot tell skipping from clipping; what only skipping can do is leave an
    # out-of-window group undecoded, and therefore unchecked for sortedness
    scrambled = copy(rows)
    scrambled[1:10, :time] = 10:-1:1          # group 1: unsorted, all < 25
    withstats = inrowgroups(joinpath(dir, "scrambled.parquet"), scrambled)
    nostats = inrowgroups(joinpath(dir, "scrambled-plain.parquet"), scrambled;
        stats = String[])
    @test nrow(load(ctx, readparquet(withstats; backend = :parquet2))) == 51
    @test_throws ArgumentError load(ctx,
        readparquet(nostats; backend = :parquet2))
    # an opaque time function is unskippable too, so it sees the bad group
    @test_throws ArgumentError load(ctx,
        readparquet(withstats; backend = :parquet2, time = row -> row.time))

    # the error paths of the reader are the fallback's too
    unsorted = writeparquetfile(joinpath(dir, "unsorted.parquet"),
        DataFrame(time = [3, 1], x = [1, 2]))
    @test_throws ArgumentError load(Context(0, 100),
        readparquet(unsorted; backend = :parquet2))
    notime = writeparquetfile(joinpath(dir, "notime.parquet"),
        DataFrame(t = [1, 2], x = [1, 2]))
    @test_throws ArgumentError load(Context(0, 100),
        readparquet(notime; backend = :parquet2))
end

@testset "writeparquet backends agree" begin
    dir = mktempdir()
    src = writeparquetfile(joinpath(dir, "src.parquet"),
        DataFrame(time = 1:5000, bid = float.(1:5000), sym = string.(1:5000)))
    ctx = Context(0, 6000)
    p = readparquet(src) |> addcolumns(r -> (; twice = r.bid * 2))
    expected = DataFrame(load(ctx, p))

    duck = joinpath(dir, "duck.parquet")
    pq2 = joinpath(dir, "pq2.parquet")
    scan(ctx, p |> writeparquet(duck; backend = :duckdb))
    scan(ctx, p |> writeparquet(pq2; backend = :parquet2))

    # same rows, whichever backend wrote and whichever reads them back
    for file in (duck, pq2), backend in (:duckdb, :parquet2)
        @test DataFrame(load(ctx, readparquet(file; backend = backend))) == expected
    end
    @test DataFrame(load(ctx, readparquet(duck))) ==
          DataFrame(load(ctx, readparquet(pq2)))

    # pass-through and the ownership contract hold for the DuckDB sink too
    mid = joinpath(dir, "mid.parquet")
    frame = load(
        ctx,
        readparquet(src) |> writeparquet(mid; backend = :duckdb) |>
        asofjoin(readparquet(src); leftprefix = "l", rightprefix = "r"),
    )
    @test names(frame) == ["time", "l_bid", "l_sym", "r_bid", "r_sym"]
    @test names(DataFrame(load(ctx, readparquet(mid)))) == ["time", "bid", "sym"]
    @test DataFrame(load(ctx, readparquet(mid))) ==
          DataFrame(load(ctx, readparquet(src)))

    # scan, load and a fully drained stream agree through the DuckDB sink
    viaload = joinpath(dir, "viaload.parquet")
    load(ctx, p |> writeparquet(viaload; backend = :duckdb))
    viastream = joinpath(dir, "viastream.parquet")
    foreach(DataFrame,
        stream(ctx, p |> writeparquet(viastream; backend = :duckdb)))
    @test DataFrame(load(ctx, readparquet(viaload))) == expected
    @test DataFrame(load(ctx, readparquet(viastream))) == expected

    # re-running truncates rather than appending
    scan(ctx, p |> writeparquet(duck; backend = :duckdb))
    @test DataFrame(load(ctx, readparquet(duck))) == expected

    # rowgroupsize under Parquet2: chunks are merged until the target is met and
    # never split, so every group but the last holds at least that many rows
    small = joinpath(dir, "small.parquet")
    scan(ctx, p |> writeparquet(small; backend = :parquet2, rowgroupsize = 1000))
    smallds = Parquet2.Dataset(small)
    groups = [nrow(DataFrame(rg)) for rg in smallds]
    @test length(groups) > 1
    @test all(>=(1000), groups[1:(end-1)])
    @test sum(groups) == nrow(expected)

    # under DuckDB it is a hint: it rounds up to its own 2048-row vector size
    ducksmall = joinpath(dir, "ducksmall.parquet")
    scan(ctx, p |> writeparquet(ducksmall; backend = :duckdb, rowgroupsize = 2048))
    @test Parquet2.nrowgroups(Parquet2.Dataset(ducksmall)) > 1

    # a stream with no rows still leaves a valid, empty file
    none = joinpath(dir, "none.parquet")
    scan(ctx, emptyframe() |> writeparquet(none; backend = :duckdb))
    @test isfile(none)
    @test nrow(DataFrame(Parquet2.Dataset(none))) == 0

    # DuckDB takes compression_codec and rejects the Parquet2-only options
    zstd = joinpath(dir, "zstd.parquet")
    scan(ctx, p |> writeparquet(zstd; backend = :duckdb, compression_codec = :zstd))
    @test DataFrame(load(ctx, readparquet(zstd))) == expected
    @test_throws ArgumentError scan(ctx,
        p |> writeparquet(joinpath(dir, "bad.parquet"); backend = :duckdb,
            npages = 3))

    # files written by either backend carry the time statistics the Parquet2
    # reader skips by, so a narrow window reads only part of them
    narrow = Context(2000, 2500)
    @test length(collect(stream(narrow, readparquet(ducksmall;
        backend = :parquet2)))) <
          Parquet2.nrowgroups(Parquet2.Dataset(ducksmall))
end

@testset "parquet backend selection" begin
    dir = mktempdir()
    path = writeparquetfile(joinpath(dir, "t.parquet"),
        DataFrame(time = 1:10, x = float.(1:10)))

    @test_throws ArgumentError readparquet(path; backend = :nope)
    @test_throws ArgumentError writeparquet(joinpath(dir, "o.parquet");
        backend = :nope)
    # both backends are loaded here, so either may be named outright
    for backend in (:auto, :duckdb, :parquet2)
        @test nrow(load(Context(0, 100), readparquet(path; backend = backend))) == 10
    end
end

@testset "writeparquet" begin
    dir = mktempdir()
    src = writeparquetfile(joinpath(dir, "ticks.parquet"),
        DataFrame(time = [1, 3, 5, 7], bid = [10.0, 10.5, 9.0, 9.5],
            ask = [11.0, 11.5, 10.0, 10.5]))
    ctx = Context(0, 100)
    p = readparquet(src) |> addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

    # pass-through: the frame is unchanged, and the file round-trips back
    out = joinpath(dir, "out.parquet")
    @test DataFrame(load(ctx, p |> writeparquet(out))) == DataFrame(load(ctx, p))
    @test DataFrame(load(ctx, readparquet(out))) == DataFrame(load(ctx, p))

    # uncurried form
    out2 = joinpath(dir, "out2.parquet")
    scan(ctx, writeparquet(p, out2))
    @test DataFrame(load(ctx, readparquet(out2))) == DataFrame(load(ctx, p))

    # multi-chunk: order preserved, one row group per rowgroupsize rows
    many = joinpath(dir, "many.parquet")
    scan(Context(0, 10), clock(1; batchsize = 2) |> writeparquet(many;
        rowgroupsize = 3))
    @test DataFrame(load(Context(0, 10), readparquet(many))).time == 0:9
    @test Parquet2.nrowgroups(Parquet2.Dataset(many)) == 3

    # the default rowgroupsize keeps a small stream in one row group
    single = joinpath(dir, "single.parquet")
    scan(Context(0, 10), clock(1; batchsize = 2) |> writeparquet(single))
    @test Parquet2.nrowgroups(Parquet2.Dataset(single)) == 1

    # rowgroupsize = 1 writes one row group per incoming chunk
    perchunk = joinpath(dir, "perchunk.parquet")
    scan(
        Context(0, 10),
        clock(1; batchsize = 2) |> writeparquet(perchunk;
            rowgroupsize = 1),
    )
    @test Parquet2.nrowgroups(Parquet2.Dataset(perchunk)) == 5

    # scan, load and a fully drained stream all write the same rows
    manyload = joinpath(dir, "manyload.parquet")
    load(Context(0, 10), clock(1; batchsize = 2) |> writeparquet(manyload))
    manystream = joinpath(dir, "manystream.parquet")
    foreach(DataFrame,
        stream(Context(0, 10), clock(1; batchsize = 2) |>
                               writeparquet(manystream)))
    expected = DataFrame(load(Context(0, 10), readparquet(many)))
    @test DataFrame(load(Context(0, 10), readparquet(manyload))) == expected
    @test DataFrame(load(Context(0, 10), readparquet(manystream))) == expected

    # re-running truncates rather than appending
    scan(Context(0, 10), clock(1; batchsize = 2) |> writeparquet(many;
        rowgroupsize = 3))
    @test DataFrame(load(Context(0, 10), readparquet(many))) == expected

    # a stream with no rows yields a valid, column-less file
    none = joinpath(dir, "none.parquet")
    scan(ctx, emptyframe() |> writeparquet(none))
    @test isfile(none)
    @test nrow(DataFrame(Parquet2.Dataset(none))) == 0

    # ownership: the writer reads its chunk on another task while downstream
    # ops mutate their own chunk's column index in place (asofjoin's
    # leftprefix), so the file must hold the unprefixed columns
    mid = joinpath(dir, "mid.parquet")
    frame = load(
        ctx,
        readparquet(src) |> writeparquet(mid) |>
        asofjoin(readparquet(src); leftprefix = "l", rightprefix = "r"),
    )
    @test names(frame) == ["time", "l_bid", "l_ask", "r_bid", "r_ask"]
    @test names(DataFrame(load(ctx, readparquet(mid)))) == ["time", "bid", "ask"]
    @test DataFrame(load(ctx, readparquet(mid))) ==
          DataFrame(load(ctx, readparquet(src)))

    # the same, against addrollingcolumns' empty-window assembly, which adds
    # columns to the incoming chunk in place
    roll = joinpath(dir, "roll.parquet")
    frame = load(
        Context(0, 10),
        clock(1; batchsize = 2) |> writeparquet(roll) |>
        addrollingcolumns((w2 = 2,), Count(); from = emptyframe()),
    )
    @test names(frame) == ["time", "w2_count"]
    @test DataFrame(load(Context(0, 10), readparquet(roll))).time == 0:9

    # forwarded Parquet2.FileWriter keywords
    zstd = joinpath(dir, "zstd.parquet")
    scan(ctx, p |> writeparquet(zstd; compression_codec = :zstd))
    @test DataFrame(load(ctx, readparquet(zstd))) == DataFrame(load(ctx, p))

    # queue = 0 (rendezvous hand-off) writes the same rows
    rendez = joinpath(dir, "rendez.parquet")
    scan(ctx, p |> writeparquet(rendez; queue = 0))
    @test DataFrame(load(ctx, readparquet(rendez))) == DataFrame(load(ctx, p))

    @test_throws ArgumentError writeparquet(out; queue = -1)
    @test_throws ArgumentError writeparquet(out; rowgroupsize = 0)

    # columns may not change mid-stream
    changing = joinpath(dir, "changing.parquet")
    @test_throws ArgumentError scan(Context(0, 10),
        clock(1; batchsize = 2) |>
        addcolumns(r -> r.time < 4 ? (; a = 1) : (; b = 2)) |>
        writeparquet(changing))
end

@testset "parquet and CSV interoperate" begin
    dir = mktempdir()
    csvpath = joinpath(dir, "ticks.csv")
    write(csvpath, "time,bid,ask\n1,10.0,11.0\n3,10.5,11.5\n5,9.0,10.0\n")
    tt = Dict(:time => Int, :bid => Float64, :ask => Float64)
    ctx = Context(0, 100)

    # CSV -> parquet -> CSV keeps the values, and parquet keeps the types
    pq = joinpath(dir, "ticks.parquet")
    scan(ctx, readcsv(csvpath; types = tt) |> writeparquet(pq))
    back = joinpath(dir, "back.csv")
    scan(ctx, readparquet(pq) |> writecsv(back))
    @test DataFrame(load(ctx, readcsv(back; types = tt))) ==
          DataFrame(load(ctx, readcsv(csvpath; types = tt)))
    @test DataFrame(load(ctx, readparquet(pq))) ==
          DataFrame(load(ctx, readcsv(csvpath; types = tt)))
end

# Runs `script` in a fresh Julia, which is the only way to see the package with
# a backend *missing* — this process has both loaded. Serial by construction:
# concurrent Julia processes race the precompile cache.
function insubprocess(script::String)
    return read(
        `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`,
        String)
end

@testset "parquet operators without a backend" begin
    # Loading CausalFrames alone must not pull in either backend, and both
    # operators must name both packages.
    @test insubprocess("""
    using CausalFrames
    @assert Base.get_extension(CausalFrames, :CausalFramesDuckDBExt) === nothing
    @assert Base.get_extension(CausalFrames, :CausalFramesParquet2Ext) === nothing
    for f in (() -> readparquet("x.parquet"), () -> writeparquet("x.parquet"))
        try
            f()
            error("expected an ArgumentError")
        catch e
            e isa ArgumentError || rethrow()
            for hint in ("using DuckDB", "using Parquet2")
                occursin(hint, e.msg) ||
                    error("message lacks \\\$hint: \\\$(e.msg)")
            end
        end
    end
    # naming an unloaded backend says so outright
    try
        readparquet("x.parquet"; backend = :duckdb)
        error("expected an ArgumentError")
    catch e
        e isa ArgumentError || rethrow()
        occursin("not loaded", e.msg) || error("unexpected message: \\\$(e.msg)")
    end
    print("ok")
    """) == "ok"
end

@testset "parquet with only one backend loaded" begin
    # With just one package loaded, both directions must still work through it.
    roundtrip = """
    using CausalFrames, DataFrames
    dir = mktempdir()
    src = joinpath(dir, "src.parquet")
    scan(Context(0, 100), clock(1) |> addcolumns(r -> (; x = r.time * 1.5)) |>
                          writeparquet(src))
    df = DataFrame(load(Context(10, 20), readparquet(src)))
    @assert nrow(df) == 10 "got \\\$(nrow(df)) rows"
    @assert df.time == 10:19
    @assert df.x == 15.0:1.5:28.5
    print("ok")
    """
    # only Parquet2: its own writer, and its row-group reader as the fallback
    @test insubprocess("using Parquet2\n" * roundtrip) == "ok"
    # only DuckDB: its streaming reader, and its staged writer as the fallback
    @test insubprocess("using DuckDB\n" * roundtrip) == "ok"
end
