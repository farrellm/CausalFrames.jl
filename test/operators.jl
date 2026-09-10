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

@testset "reordercolumns" begin
    ctx = Context(0, 100)
    src = clock(1) |> addcolumns(
        r -> (; bid = 1.0 * r.time, ask = 2.0 * r.time, sym = "a"))

    # the matched columns move to the front, the rest follow in input order
    @test names(load(ctx, src |> reordercolumns(:sym))) ==
          ["time", "sym", "bid", "ask"]

    # ...and they move in the selectors' order, not the input's — the whole
    # difference from selectcolumns, which would give ["time", "bid", "ask"]
    @test names(load(ctx, src |> reordercolumns(:ask, :bid))) ==
          ["time", "ask", "bid", "sym"]

    # every selector form
    @test names(load(ctx, src |> reordercolumns("sym"))) ==
          ["time", "sym", "bid", "ask"]
    @test names(load(ctx, src |> reordercolumns(r"^s"))) ==
          ["time", "sym", "bid", "ask"]
    @test names(load(ctx, src |> reordercolumns(startswith("s")))) ==
          ["time", "sym", "bid", "ask"]

    # a collection is punctuation, not a group: it flattens in place
    @test names(load(ctx, src |> reordercolumns([:ask, :bid]))) ==
          names(load(ctx, src |> reordercolumns(:ask, :bid)))
    @test names(load(ctx, src |> reordercolumns([[:sym], (r"^a",)]))) ==
          ["time", "sym", "ask", "bid"]

    # a pattern says which columns, not which order: its matches keep the
    # input's order among themselves
    @test names(load(ctx, src |> reordercolumns(r"s"))) ==
          ["time", "ask", "sym", "bid"]

    # a column matched twice is placed once, by the first selector
    @test names(load(ctx, src |> reordercolumns(:sym, r"", :bid))) ==
          ["time", "sym", "bid", "ask"]

    # values ride along untouched
    df = DataFrame(load(Context(0, 4), src |> reordercolumns(:ask)))
    @test names(df) == ["time", "ask", "bid", "sym"]
    @test df.time == 0:3
    @test df.ask == [0.0, 2.0, 4.0, 6.0]
    @test df.bid == [0.0, 1.0, 2.0, 3.0]

    # :time is pinned first, and is never matched by a pattern
    @test names(load(ctx, src |> reordercolumns(r""))) ==
          ["time", "bid", "ask", "sym"]
    @test names(load(ctx, src |> reordercolumns(_ -> true))) ==
          ["time", "bid", "ask", "sym"]
    @test_throws ArgumentError reordercolumns(:time)          # eager
    @test_throws ArgumentError reordercolumns([:bid, "time"])  # eager, nested

    # naming a column the data does not have
    @test_throws ArgumentError load(ctx, src |> reordercolumns(:nope))

    # at least one selector is required, in either form
    @test_throws ArgumentError reordercolumns()
    @test_throws ArgumentError reordercolumns(src)

    # an unusable selector
    @test_throws ArgumentError load(ctx, src |> reordercolumns(1.5))

    # a reorder into the order already held passes the chunk through unchanged
    @test DataFrame(load(ctx, src |> reordercolumns(:bid, :ask, :sym))) ==
          DataFrame(load(ctx, src))
    @test DataFrame(load(ctx, src |> reordercolumns(r"zzz"))) ==
          DataFrame(load(ctx, src))

    # the uncurried, pipeline-first form is equivalent to the |> chain
    @test DataFrame(load(ctx, reordercolumns(src, :sym, :ask))) ==
          DataFrame(load(ctx, src |> reordercolumns(:sym, :ask)))

    # a no-op on an empty frame
    @test nrow(load(ctx, emptyframe() |> reordercolumns(r"x"))) == 0

    # it composes with the projections
    @test names(load(ctx, src |> dropcolumns(:bid) |> reordercolumns(:sym))) ==
          ["time", "sym", "ask"]
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

@testset "reordercolumns over several chunks" begin
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

    # the resolved order is cached per run, so it must hold across chunk
    # boundaries and be rebuilt for the next run of the same pipeline
    p = chunked |> reordercolumns(:y)
    df = DataFrame(load(ctx, p))
    @test names(df) == ["time", "y", "x"]
    @test df.y == 3 .* (1:50)
    @test df.x == 2 .* (1:50)
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

@testset "lag" begin
    ctx = Context(0, 10)
    # clock clips to the context; carry the (pre-shift) time as a value column
    src = clock(1) |> addcolumns(r -> (; v = float(r.time)))

    # lag shifts every row +offset in time; other columns pass through, so the
    # value seen at time t is the one the input had at t - offset.
    df = DataFrame(load(ctx, src |> lag(2)))
    # the window slid back to [-2, 8) fills all of [0, 10); v at output t is (t - 2)
    @test df.time == 0:9
    @test df.v == collect(-2.0:7.0)
    @test names(df) == ["time", "v"]

    # the upstream pipeline is run over the window slid back by the offset
    seen = Ref{Any}(nothing)
    recorder = CausalPipeline() do c
        seen[] = c
        [DataFrame(time = [c.start], v = [1.0])]
    end
    load(ctx, recorder |> lag(3))
    @test seen[] == Context(-3, 7)

    # offset 0 is the identity
    @test isequal(DataFrame(load(ctx, src |> lag(0))), DataFrame(load(ctx, src)))

    # a negative offset would look forward (acausal); rejected when the
    # pipeline runs, mirroring asofjoin's tolerance guard
    @test_throws ArgumentError load(ctx, src |> lag(-1))

    # streaming matches load
    streamed = reduce(vcat, [DataFrame(f) for f in stream(ctx, src |> lag(2))])
    @test isequal(streamed, df)

    # Dates time with a Period offset
    dsrc = CausalPipeline() do c
        [DataFrame(time = [DateTime(2020, 1, 1, 1), DateTime(2020, 1, 1, 2)],
                x = [1, 2])]
    end
    dctx = Context(DateTime(2020, 1, 1), DateTime(2020, 1, 2))
    ddf = DataFrame(load(dctx, dsrc |> lag(Hour(1))))
    @test ddf.time == [DateTime(2020, 1, 1, 2), DateTime(2020, 1, 1, 3)]
    @test ddf.x == [1, 2]

    # transform on an empty frame is a no-op
    @test nrow(load(ctx, emptyframe() |> lag(2))) == 0

    # the uncurried, pipeline-first form equals the |> chain
    @test isequal(DataFrame(load(ctx, lag(src, 2))), df)
end

@testset "head" begin
    ctx = Context(0, 100)
    src = clock(1; batchsize = 4)

    @test DataFrame(load(ctx, src |> head(10))).time == 0:9
    @test DataFrame(load(ctx, src |> head(6))).time == 0:5     # partial slice
    # a budget larger than the stream just takes everything
    @test DataFrame(load(Context(0, 5), src |> head(10))).time == 0:4
    blank = load(ctx, src |> head(0))
    @test nrow(blank) == 0 && names(blank) == ["time"]

    # The early-exit proof: upstream is asked for exactly the chunks head
    # consumes and not one more. Built on a chunkmap, this would drain all 25.
    pulled = Ref(0)
    counting = CausalPipeline() do _
        (
            begin
                pulled[] += 1
                DataFrame(time = collect((4i-4):(4i-1)), v = fill(i, 4))
            end for i in 1:25
        )
    end
    @test DataFrame(load(ctx, counting |> head(6))).time == 0:5
    @test pulled[] == 2                       # not 25
    pulled[] = 0                              # a budget landing on a boundary
    @test DataFrame(load(ctx, counting |> head(8))).time == 0:7
    @test pulled[] == 2                       # no extra pull to discover the end
    pulled[] = 0                              # stream's lookahead must not reach upstream
    @test reduce(vcat, DataFrame.(stream(ctx, counting |> head(8)))).time == 0:7
    @test pulled[] == 2
    pulled[] = 0                              # head(0) never advances the generator
    @test nrow(load(ctx, counting |> head(0))) == 0
    @test pulled[] == 0
    pulled[] = 0                              # baseline: without head, everything
    @test nrow(load(ctx, counting)) == 100 && pulled[] == 25

    # the upstream context is not rewritten
    seen = Ref{Any}(nothing)
    recorder = CausalPipeline() do c
        seen[] = c
        [DataFrame(time = [c.start])]
    end
    load(ctx, recorder |> head(1))
    @test seen[] == ctx

    # a whole-chunk take shares the chunk rather than copying it
    chunk = DataFrame(time = [1, 2])
    frame = load(ctx, CausalPipeline(_ -> [chunk]) |> head(5))
    @test only(frame.chunks) === chunk

    @test_throws ArgumentError head(-1)                        # eager
    @test nrow(load(ctx, emptyframe() |> head(3))) == 0        # empty frame no-op
    # the uncurried, pipeline-first form equals the |> chain
    @test isequal(DataFrame(load(ctx, head(src, 6))),
        DataFrame(load(ctx, src |> head(6))))
    # streaming matches load
    @test isequal(reduce(vcat, DataFrame.(stream(ctx, src |> head(6)))),
        DataFrame(load(ctx, src |> head(6))))
end

@testset "settime" begin
    ctx = Context(0, 10)
    src = clock(1; batchsize = 3) |> addcolumns(r -> (; v = float(r.time)))

    # the function form overwrites :time in place, keeping its position
    df = DataFrame(load(ctx, src |> settime(r -> r.time + 2)))
    @test df.time == 2:9          # 8 and 9 shifted to 10, 11 and were clipped
    @test df.v == collect(0.0:7.0)
    @test names(df) == ["time", "v"]

    # the symbol form makes that column :time, in its own position, and the old
    # :time column disappears
    sdf = DataFrame(load(ctx, src |> addcolumns(r -> (; t2 = r.time + 2)) |>
                              settime(:t2)))
    @test names(sdf) == ["v", "time"]
    @test sdf.time == 2:9
    @test sdf.v == collect(0.0:7.0)

    # settime cannot widen the window the way lag does, so the rows lag pulls in
    # from before start are simply absent
    @test DataFrame(load(ctx, src |> settime(r -> r.time + 2))).time == 2:9
    @test DataFrame(load(ctx, src |> lag(2))).time == 0:9
    seen = Ref{Any}(nothing)
    recorder = CausalPipeline() do c
        seen[] = c
        [DataFrame(time = [c.start])]
    end
    load(ctx, recorder |> settime(r -> r.time))
    @test seen[] == ctx

    # settime(:time) leaves the values alone but still re-clips to [start, stop),
    # so summarize's row sitting exactly at stop is dropped
    @test nrow(load(ctx, src |> summarize(Count()))) == 1
    @test nrow(load(ctx, src |> summarize(Count()) |> settime(:time))) == 0
    @test isequal(DataFrame(load(ctx, src |> settime(:time))),
        DataFrame(load(ctx, src)))

    # a row may not move earlier in time
    @test_throws ArgumentError load(ctx, src |> settime(r -> r.time - 1))
    # nor may the result be out of order within a chunk
    @test_throws ArgumentError load(ctx, src |> settime(r -> 9 - r.time))
    # ... nor across a chunk boundary, which the forward and within-chunk rules
    # do not imply: every row here moves forward and each chunk is sorted, yet
    # the stream emits 5, 9, 6, 7. This is why settimechunk! carries prevtime.
    twochunks = CausalPipeline() do _
        [DataFrame(time = [1, 2], x = [5, 9]), DataFrame(time = [3, 4], x = [6, 7])]
    end
    @test_throws ArgumentError load(ctx, twochunks |> settime(:x))
    # a textual column cannot be ordered against the window
    @test_throws ArgumentError load(ctx,
        src |> addcolumns(r -> (; s = "x")) |> settime(:s))
    # and the named column must exist
    @test_throws ArgumentError load(ctx, src |> settime(:nope))

    # eager: the spec must be a column name or a per-row function
    @test_throws ArgumentError settime(3)
    @test_throws ArgumentError settime(src)

    # multi-chunk state: prevtime is carried, and a legal shift streams cleanly
    shifted = src |> settime(r -> r.time + 1)
    @test isequal(reduce(vcat, DataFrame.(stream(ctx, shifted))),
        DataFrame(load(ctx, shifted)))

    # transform on an empty frame is a no-op
    @test nrow(load(ctx, emptyframe() |> settime(r -> r.time))) == 0
    # the uncurried, pipeline-first form equals the |> chain
    @test isequal(DataFrame(load(ctx, settime(src, r -> r.time + 2))), df)
end

@testset "concatenate" begin
    ctx = Context(0, 10)
    withv = addcolumns(r -> (; v = float(r.time)))
    whole = clock(1) |> withv
    early = clock(1) |> filterrows(r -> r.time < 4) |> withv
    late = clock(1) |> filterrows(r -> r.time >= 4) |> withv

    # the pipelines are emitted end to end, so splitting a stream in time and
    # concatenating the halves reproduces it
    df = DataFrame(load(ctx, concatenate(early, late)))
    @test df.time == 0:9
    @test df.v == collect(0.0:9.0)
    @test names(df) == ["time", "v"]
    @test isequal(df, DataFrame(load(ctx, whole)))

    # every pipeline is run over the full context, and clips itself
    seen = Context[]
    recorder(t) = CausalPipeline() do c
        push!(seen, c)
        return [DataFrame(time = [t], x = [t])]
    end
    load(ctx, concatenate(recorder(1), recorder(2)))
    @test seen == [ctx, ctx]

    # a pipeline is not run until the previous one is exhausted
    runs = Int[]
    probe(i, t) = CausalPipeline() do c
        push!(runs, i)
        return [DataFrame(time = [t], x = [i])]
    end
    it = concatenate(probe(1, 1), probe(2, 2)).run(ctx)
    next = iterate(it)
    @test runs == [1]
    @test iterate(it, next[2]) !== nothing
    @test runs == [1, 2]

    at(t, x) = CausalPipeline(c -> [DataFrame(time = [t], x = [x])])

    # equal times across a boundary are fine — the output only has to be
    # non-decreasing
    @test DataFrame(load(ctx, concatenate(at(5, 1), at(5, 2)))).x == [1, 2]

    # pipelines out of time order are rejected
    @test_throws ArgumentError load(ctx, concatenate(at(5, 1), at(4, 2)))
    @test_throws ArgumentError load(ctx, concatenate(late, early))

    # so are differing columns, whether renamed or merely reordered
    ab = CausalPipeline(c -> [DataFrame(time = [1], a = [1], b = [2])])
    ba = CausalPipeline(c -> [DataFrame(time = [2], b = [2], a = [1])])
    ac = CausalPipeline(c -> [DataFrame(time = [2], a = [1], c = [3])])
    @test_throws ArgumentError load(ctx, concatenate(ab, ba))
    @test_throws ArgumentError load(ctx, concatenate(ab, ac))

    # element types promote across pipelines, as they do across chunks
    ints = CausalPipeline(c -> [DataFrame(time = [1], x = [1])])
    floats = CausalPipeline(c -> [DataFrame(time = [2], x = [2.5])])
    @test eltype(DataFrame(load(ctx, concatenate(ints, floats))).x) == Float64

    # pipelines producing nothing are skipped, wherever they sit
    @test isequal(
        DataFrame(
            load(ctx, concatenate(emptyframe(), early, emptyframe(), late,
                emptyframe())),
        ),
        df,
    )

    # no pipelines at all is emptyframe, the identity of concatenation
    blank = load(ctx, concatenate())
    @test nrow(blank) == 0
    @test names(blank) == ["time"]

    # one pipeline is that pipeline
    @test isequal(DataFrame(load(ctx, concatenate(whole))),
        DataFrame(load(ctx, whole)))

    # the result chains like any other pipeline
    @test DataFrame(load(ctx,
        concatenate(early, late) |> filterrows(r -> r.v > 6))).time == 7:9

    # streaming matches load
    streamed = reduce(vcat,
        [DataFrame(f) for f in stream(ctx, concatenate(early, late))])
    @test isequal(streamed, df)
end
