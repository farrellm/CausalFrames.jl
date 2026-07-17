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
    write(path, """
        time,bid,ask
        1,10.0,11.0
        3,10.5,11.5
        5,9.0,10.0
        7,9.5,10.5
        """)

    frame = load(Context(0, 100), readcsv(path))
    @test nrow(frame) == 4
    @test names(frame) == ["time", "bid", "ask"]

    # clipped to [start, stop)
    frame = load(Context(3, 7), readcsv(path))
    @test DataFrame(frame).time == [3, 5]

    # time column converted to the context's time type
    frame = load(Context(0.0, 100.0), readcsv(path))
    @test eltype(DataFrame(frame).time) == Float64

    unsorted = joinpath(mktempdir(), "unsorted.csv")
    write(unsorted, "time,x\n3,1\n1,2\n")
    @test_throws ArgumentError load(Context(0, 100), readcsv(unsorted))

    notime = joinpath(mktempdir(), "notime.csv")
    write(notime, "t,x\n1,2\n")
    @test_throws ArgumentError load(Context(0, 100), readcsv(notime))
end

@testset "filterrows and addcolumns" begin
    path = joinpath(mktempdir(), "ticks.csv")
    write(path, """
        time,bid,ask
        1,10.0,11.0
        3,10.5,11.5
        5,9.0,10.0
        7,9.5,10.5
        """)

    p = readcsv(path) |>
        filterrows(r -> r.bid >= 9.5) |>
        addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
    df = DataFrame(load(Context(0, 100), p))
    @test df.time == [1, 3, 7]
    @test df.mid == [10.5, 11.0, 10.0]
    @test names(df) == ["time", "bid", "ask", "mid"]

    # row access by symbol and access to :time
    p = readcsv(path) |> filterrows(r -> r[:time] > 3)
    @test DataFrame(load(Context(0, 100), p)).time == [5, 7]

    # addcolumns may not touch time, and must return a NamedTuple
    bad = readcsv(path) |> addcolumns(r -> (; time = r.time))
    @test_throws ArgumentError load(Context(0, 100), bad)
    notuple = readcsv(path) |> addcolumns(r -> r.bid)
    @test_throws ArgumentError load(Context(0, 100), notuple)

    # transforms on an empty frame are no-ops
    p = emptyframe() |> filterrows(r -> true) |> addcolumns(r -> (; y = 1))
    @test nrow(load(Context(0, 100), p)) == 0
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

    # small chunks stream as several frames but load identically
    p = readcsv(path; chunkbytes = 64)
    frames = collect(stream(Context(0, 100), p))
    @test length(frames) > 1
    @test DataFrame(load(Context(0, 100), p)) ==
          DataFrame(load(Context(0, 100), readcsv(path)))
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
        readcsv(badtail; chunkbytes = 64))).time == 1:9
    # ... but reading through it throws
    @test_throws ArgumentError load(Context(0, 1000),
        readcsv(badtail; chunkbytes = 64))

    @test_throws ArgumentError readcsv(path; chunkbytes = 0)
end
