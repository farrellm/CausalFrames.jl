@testset "clock" begin
    frame = load(Context(0, 10), clock(3))
    @test DataFrame(frame).time == [0, 3, 6, 9]

    # stop is excluded (half-open interval)
    frame = load(Context(0, 9), clock(3))
    @test DataFrame(frame).time == [0, 3, 6]

    frame = load(Context(DateTime(2026, 1, 1), DateTime(2026, 1, 1, 1)),
        clock(Minute(15)))
    @test nrow(frame) == 4
    @test DataFrame(frame).time[end] == DateTime(2026, 1, 1, 0, 45)

    @test nrow(load(Context(5, 5), clock(1))) == 0
    @test_throws ArgumentError load(Context(0, 10), clock(0))
    @test_throws ArgumentError load(Context(0, 10), clock(-1))
end

@testset "readcsv" begin
    path = joinpath(mktempdir(), "ticks.csv")
    write(
        path,
        """
time,bid,ask
1,10.0,11.0
3,10.5,11.5
5,9.0,10.0
7,9.5,10.5
""",
    )
    tt = Dict(:time => Int, :bid => Float64, :ask => Float64)

    frame = load(Context(0, 100), readcsv(path; types = tt))
    @test nrow(frame) == 4
    @test names(frame) == ["time", "bid", "ask"]

    # clipped to [start, stop)
    frame = load(Context(3, 7), readcsv(path; types = tt))
    @test DataFrame(frame).time == [3, 5]

    # time column converted to the context's time type
    frame = load(Context(0.0, 100.0), readcsv(path; types = tt))
    @test eltype(DataFrame(frame).time) == Float64

    # untyped columns stay String (no type inference)
    df = DataFrame(load(Context(0, 100), readcsv(path; types = Dict(:time => Int))))
    @test eltype(df.bid) == String
    @test df.bid == ["10.0", "10.5", "9.0", "9.5"]

    unsorted = joinpath(mktempdir(), "unsorted.csv")
    write(unsorted, "time,x\n3,1\n1,2\n")
    @test_throws ArgumentError load(
        Context(0, 100), readcsv(unsorted; types = Dict(:time => Int)))

    notime = joinpath(mktempdir(), "notime.csv")
    write(notime, "t,x\n1,2\n")
    @test_throws ArgumentError load(
        Context(0, 100), readcsv(notime; types = Dict(:time => Int)))

    # the time column must be typed (or produced by a function), else error
    @test_throws ArgumentError readcsv(path)                       # eager
    @test_throws ArgumentError readcsv(path; types = Dict(:bid => Float64))
    # a positional `types` that misses the time column: caught on first chunk
    @test_throws ArgumentError load(
        Context(0, 100), readcsv(path; types = [String, Float64, Float64]))
end

@testset "readcsv time selection, rename, delim" begin
    dir = mktempdir()

    # time as a Symbol: name the time column, typed via `types`
    path = joinpath(dir, "ts.csv")
    write(path, "ts,x\n1,a\n2,b\n3,c\n")
    frame = load(Context(0, 100), readcsv(path; time = :ts,
        types = Dict(:ts => Int)))
    @test names(frame) == ["time", "x"]   # ts renamed to time
    df = DataFrame(frame)
    @test df.time == [1, 2, 3]
    @test df.x == ["a", "b", "c"]

    # a Symbol time column still needs a type, else error
    @test_throws ArgumentError readcsv(path; time = :ts)

    # time as a per-row function: produces the :time column from strings
    path2 = joinpath(dir, "func.csv")
    write(path2, "stamp,x\n001,a\n002,b\n003,c\n")
    frame = load(Context(0, 100),
        readcsv(path2; time = row -> parse(Int, row.stamp)))
    df = DataFrame(frame)
    @test eltype(df.time) == Int
    @test df.time == [1, 2, 3]

    # rename (map), applied before the time name is resolved. `types` names
    # the original column `:t` (typing happens before the rename).
    path3 = joinpath(dir, "rename.csv")
    write(path3, "t,v\n1,a\n2,b\n")
    frame = load(
        Context(0, 100),
        readcsv(path3; rename = Dict("t" => "time"), types = Dict(:t => Int)),
    )
    @test DataFrame(frame).time == [1, 2]

    # rename (function), applied before the time function reads renamed names
    frame = load(
        Context(0, 100),
        readcsv(path3;
            rename = n -> uppercase(String(n)), time = row -> parse(Int, row.T)),
    )
    df = DataFrame(frame)
    @test names(df) == ["T", "V", "time"]
    @test df.time == [1, 2]

    # delim
    path4 = joinpath(dir, "semi.csv")
    write(path4, "time;x\n1;a\n2;b\n")
    frame = load(Context(0, 100), readcsv(path4; delim = ';',
        types = Dict(:time => Int)))
    df = DataFrame(frame)
    @test df.time == [1, 2]
    @test df.x == ["a", "b"]
end

@testset "writecsv" begin
    dir = mktempdir()
    src = joinpath(dir, "ticks.csv")
    write(
        src,
        """
time,bid,ask
1,10.0,11.0
3,10.5,11.5
5,9.0,10.0
7,9.5,10.5
""",
    )
    tt = Dict(:time => Int, :bid => Float64, :ask => Float64)
    ctx = Context(0, 100)
    p = readcsv(src; types = tt) |> addcolumns(r -> (; mid = (r.bid + r.ask) / 2))

    # pass-through: the frame is unchanged, and the file round-trips back
    out = joinpath(dir, "out.csv")
    @test DataFrame(load(ctx, p |> writecsv(out))) == DataFrame(load(ctx, p))
    back = Dict(:time => Int, :bid => Float64, :ask => Float64, :mid => Float64)
    @test DataFrame(load(ctx, readcsv(out; types = back))) == DataFrame(load(ctx, p))

    # uncurried form
    out2 = joinpath(dir, "out2.csv")
    scan(ctx, writecsv(p, out2))
    @test read(out2, String) == read(out, String)

    # multi-chunk: the header is written exactly once, order preserved
    many = joinpath(dir, "many.csv")
    scan(Context(0, 10), clock(1; batchsize = 2) |> writecsv(many))
    lines = readlines(many)
    @test lines[1] == "time"
    @test count(==("time"), lines) == 1
    @test parse.(Int, lines[2:end]) == 0:9

    # scan and load write identical files
    manyload = joinpath(dir, "manyload.csv")
    load(Context(0, 10), clock(1; batchsize = 2) |> writecsv(manyload))
    @test read(manyload, String) == read(many, String)

    # a fully drained stream writes the same file too
    manystream = joinpath(dir, "manystream.csv")
    foreach(
        DataFrame,
        stream(Context(0, 10), clock(1; batchsize = 2) |>
                               writecsv(manystream)),
    )
    @test read(manystream, String) == read(many, String)

    # re-running truncates rather than appending
    scan(Context(0, 10), clock(1; batchsize = 2) |> writecsv(many))
    @test readlines(many) == lines

    # a stream with no rows yields an empty file
    none = joinpath(dir, "none.csv")
    scan(ctx, emptyframe() |> writecsv(none))
    @test isfile(none) && isempty(read(none, String))

    # ownership: the writer reads its chunk on another task while downstream
    # ops mutate their own chunk's column index in place (asofjoin's
    # leftprefix), so the file must hold the unprefixed columns
    mid = joinpath(dir, "mid.csv")
    frame = load(
        ctx,
        readcsv(src; types = tt) |> writecsv(mid) |>
        asofjoin(readcsv(src; types = tt); leftprefix = "l", rightprefix = "r"),
    )
    @test names(frame) == ["time", "l_bid", "l_ask", "r_bid", "r_ask"]
    @test readlines(mid)[1] == "time,bid,ask"
    @test DataFrame(load(ctx, readcsv(mid; types = tt))) ==
          DataFrame(load(ctx, readcsv(src; types = tt)))

    # the same, against addrollingcolumns' empty-window assembly, which adds
    # columns to the incoming chunk in place
    roll = joinpath(dir, "roll.csv")
    frame = load(
        Context(0, 10),
        clock(1; batchsize = 2) |> writecsv(roll) |>
        addrollingcolumns((w2 = 2,), Count(); from = emptyframe()),
    )
    @test names(frame) == ["time", "w2_count"]
    @test readlines(roll) == vcat("time", string.(0:9))

    # forwarded CSV.write keywords
    tabbed = joinpath(dir, "tabbed.csv")
    scan(ctx, p |> writecsv(tabbed; delim = '\t'))
    @test readlines(tabbed)[1] == "time\tbid\task\tmid"

    # queue = 0 (rendezvous hand-off) writes the same file
    rendez = joinpath(dir, "rendez.csv")
    scan(ctx, p |> writecsv(rendez; queue = 0))
    @test read(rendez, String) == read(out, String)

    # the keywords writecsv controls itself are rejected eagerly
    for k in (:append, :header, :writeheader, :partition, :compress)
        @test_throws ArgumentError writecsv(out; (k => true,)...)
    end
    @test_throws ArgumentError writecsv(out; queue = -1)

    # a writer failure propagates to the consumer
    bad = joinpath(dir, "nosuchdir", "out.csv")
    @test_throws Exception scan(ctx, p |> writecsv(bad))
end

@testset "filterrows and addcolumns" begin
    path = joinpath(mktempdir(), "ticks.csv")
    write(
        path,
        """
time,bid,ask
1,10.0,11.0
3,10.5,11.5
5,9.0,10.0
7,9.5,10.5
""",
    )
    tt = Dict(:time => Int, :bid => Float64, :ask => Float64)

    p =
        readcsv(path; types = tt) |>
        filterrows(r -> r.bid >= 9.5) |>
        addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
    df = DataFrame(load(Context(0, 100), p))
    @test df.time == [1, 3, 7]
    @test df.mid == [10.5, 11.0, 10.0]
    @test names(df) == ["time", "bid", "ask", "mid"]

    # row access by symbol and access to :time
    p = readcsv(path; types = tt) |> filterrows(r -> r[:time] > 3)
    @test DataFrame(load(Context(0, 100), p)).time == [5, 7]

    # addcolumns may not touch time, and must return a NamedTuple
    bad = readcsv(path; types = tt) |> addcolumns(r -> (; time = r.time))
    @test_throws ArgumentError load(Context(0, 100), bad)
    notuple = readcsv(path; types = tt) |> addcolumns(r -> r.bid)
    @test_throws ArgumentError load(Context(0, 100), notuple)

    # transforms on an empty frame are no-ops
    p = emptyframe() |> filterrows(r -> true) |> addcolumns(r -> (; y = 1))
    @test nrow(load(Context(0, 100), p)) == 0

    # the uncurried, pipeline-first forms are equivalent to the |> chain
    src = readcsv(path; types = tt)
    curried =
        src |> filterrows(r -> r.bid >= 9.5) |>
        addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
    uncurried = addcolumns(filterrows(src, r -> r.bid >= 9.5),
        r -> (; mid = (r.bid + r.ask) / 2))
    @test DataFrame(load(Context(0, 100), curried)) ==
          DataFrame(load(Context(0, 100), uncurried))
end

@testset "selectcolumns and dropcolumns" begin
    ctx = Context(0, 100)
    src = clock(1) |> addcolumns(
        r -> (; bid = 1.0 * r.time, ask = 2.0 * r.time, sym = "a"))

    # every selector form, on both operators
    @test names(load(ctx, src |> selectcolumns(:bid))) == ["time", "bid"]
    @test names(load(ctx, src |> selectcolumns("bid"))) == ["time", "bid"]
    @test names(load(ctx, src |> selectcolumns(r"^.s"))) == ["time", "ask"]
    @test names(load(ctx, src |> selectcolumns(startswith("b")))) ==
          ["time", "bid"]
    @test names(load(ctx, src |> dropcolumns(:sym))) == ["time", "bid", "ask"]
    @test names(load(ctx, src |> dropcolumns(endswith("id")))) ==
          ["time", "ask", "sym"]

    # varargs, and collections nested arbitrarily deep
    @test names(load(ctx, src |> selectcolumns(:bid, r"sym"))) ==
          ["time", "bid", "sym"]
    @test names(load(ctx, src |> selectcolumns([[:bid], (r"sym",)]))) ==
          ["time", "bid", "sym"]
    @test names(load(ctx, src |> dropcolumns([:bid, "ask"]))) == ["time", "sym"]

    # output keeps the input's column order, not the selectors'
    @test names(load(ctx, src |> selectcolumns(:sym, :bid))) ==
          ["time", "bid", "sym"]

    # values ride along untouched
    df = DataFrame(load(Context(0, 4), src |> selectcolumns(:ask)))
    @test df.time == 0:3
    @test df.ask == [0.0, 2.0, 4.0, 6.0]

    # :time survives everything, and is never matched by a pattern
    @test names(load(ctx, src |> dropcolumns(r""))) == ["time"]
    @test names(load(ctx, src |> dropcolumns(_ -> true))) == ["time"]
    @test names(load(ctx, src |> selectcolumns(:time))) == ["time"]
    @test_throws ArgumentError dropcolumns(:time)          # eager
    @test_throws ArgumentError dropcolumns([:bid, "time"])  # eager, nested

    # naming a column the data does not have
    @test_throws ArgumentError load(ctx, src |> selectcolumns(:nope))
    @test_throws ArgumentError load(ctx, src |> dropcolumns(:nope))

    # at least one selector is required, in either form
    @test_throws ArgumentError selectcolumns()
    @test_throws ArgumentError dropcolumns()
    @test_throws ArgumentError selectcolumns(src)
    @test_throws ArgumentError dropcolumns(src)

    # an unusable selector
    @test_throws ArgumentError load(ctx, src |> selectcolumns(1.5))

    # selecting everything passes the chunk through unchanged
    @test DataFrame(load(ctx, src |> selectcolumns(r""))) ==
          DataFrame(load(ctx, src))
    @test DataFrame(load(ctx, src |> dropcolumns(r"zzz"))) ==
          DataFrame(load(ctx, src))

    # the uncurried, pipeline-first forms are equivalent to the |> chain
    @test DataFrame(load(ctx, selectcolumns(src, :bid, :ask))) ==
          DataFrame(load(ctx, src |> selectcolumns(:bid, :ask)))
    @test DataFrame(load(ctx, dropcolumns(src, :sym))) ==
          DataFrame(load(ctx, src |> dropcolumns(:sym)))

    # a no-op on an empty frame
    @test nrow(load(ctx, emptyframe() |> selectcolumns(r"x"))) == 0
end

@testset "selectcolumns over several chunks" begin
    path = joinpath(mktempdir(), "long.csv")
    open(path, "w") do io
        println(io, "time,x,y")
        for t in 1:50
            println(io, "$t,$(2t),$(3t)")
        end
    end
    types = Dict(:time => Int, :x => Int, :y => Int)
    ctx = Context(0, 100)

    chunked = readcsv(path; types = types, chunkbytes = 64)
    @test length(collect(stream(ctx, chunked))) > 1

    # the resolution is cached per run, so it must hold across chunk
    # boundaries and be rebuilt for the next run of the same pipeline
    p = chunked |> dropcolumns(:x)
    df = DataFrame(load(ctx, p))
    @test names(df) == ["time", "y"]
    @test df.y == 3 .* (1:50)
    @test DataFrame(load(ctx, p)) == df
    @test reduce(vcat, DataFrame.(stream(ctx, p))) == df
end

@testset "readcsv chunked" begin
    dir = mktempdir()
    path = joinpath(dir, "long.csv")
    open(path, "w") do io
        println(io, "time,x")
        for t in 1:50
            println(io, "$t,$(2t)")
        end
    end

    it = Dict(:time => Int)

    # small chunks stream as several frames but load identically
    p = readcsv(path; types = it, chunkbytes = 64)
    frames = collect(stream(Context(0, 100), p))
    @test length(frames) > 1
    @test DataFrame(load(Context(0, 100), p)) ==
          DataFrame(load(Context(0, 100), readcsv(path; types = it)))
    @test DataFrame(load(Context(0, 100), p)).time == 1:50

    # clipping to [start, stop) works across chunk boundaries
    @test DataFrame(load(Context(10, 20), p)).time == 10:19

    # early stop: disorder past the window is never read ...
    badtail = joinpath(dir, "badtail.csv")
    open(badtail, "w") do io
        println(io, "time,x")
        for t in 1:50
            println(io, "$t,$(2t)")
        end
        println(io, "7,0")   # unsorted, far past stop below
    end
    @test DataFrame(load(Context(0, 10),
        readcsv(badtail; types = it, chunkbytes = 64))).time == 1:9
    # ... but reading through it throws
    @test_throws ArgumentError load(Context(0, 1000),
        readcsv(badtail; types = it, chunkbytes = 64))

    @test_throws ArgumentError readcsv(path; types = it, chunkbytes = 0)
end
