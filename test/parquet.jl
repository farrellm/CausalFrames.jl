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

@testset "parquet operators without their backends" begin
    # Loading CausalFrames alone must not pull in the backends, and each
    # operator must say which package to load. Checked in a subprocess, since
    # this process has both loaded. Run serially: concurrent Julia processes
    # race the precompile cache.
    script = """
    using CausalFrames
    @assert Base.get_extension(CausalFrames, :CausalFramesDuckDBExt) === nothing
    @assert Base.get_extension(CausalFrames, :CausalFramesParquet2Ext) === nothing
    for (f, hint) in ((() -> readparquet("x.parquet"), "using DuckDB"),
        (() -> writeparquet("x.parquet"), "using Parquet2"))
        try
            f()
            error("expected an ArgumentError")
        catch e
            e isa ArgumentError || rethrow()
            occursin(hint, e.msg) || error("message lacks \\\$hint: \\\$(e.msg)")
        end
    end
    print("ok")
    """
    out = read(
        `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`,
        String)
    @test out == "ok"
end
