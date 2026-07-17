@testset "summarize" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(path, """
        time,sym,qty
        1,a,10
        2,b,20
        2,a,30
        4,a,40
        """)

    # no key: one row at ctx.stop, input columns dropped
    frame = load(Context(0, 9), readcsv(path) |> summarize([Count(), Sum(:qty)]))
    df = DataFrame(frame)
    @test names(df) == ["time", "count", "qty_sum"]
    @test df.time == [9]
    @test df.count == [4]
    @test df.qty_sum == [100]

    # a single summarizer (not a collection) works too
    df = DataFrame(load(Context(0, 9), readcsv(path) |> summarize(Count())))
    @test df.count == [4]

    # multi-valued summarizer
    df = DataFrame(load(Context(0, 9), readcsv(path) |> summarize(MinMax(:qty))))
    @test names(df) == ["time", "qty_min", "qty_max"]
    @test df.qty_min == [10]
    @test df.qty_max == [40]

    # duplicate configurations dedup to one column
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize([Sum(:qty), Sum(:qty)])))
    @test names(df) == ["time", "qty_sum"]

    # min, max, first, last
    df = DataFrame(load(Context(0, 9), readcsv(path) |>
        summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)])))
    @test names(df) == ["time", "qty_min", "qty_max", "qty_first", "qty_last"]
    @test df.qty_min == [10]
    @test df.qty_max == [40]
    @test df.qty_first == [10]
    @test df.qty_last == [40]

    # sum of elements to the Nth power; :x_sum2 never dedups against :x_sum
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize([Sum(:qty), SumPower(:qty, 2)])))
    @test names(df) == ["time", "qty_sum", "qty_sum2"]
    @test df.qty_sum == [100]
    @test df.qty_sum2 == [3000]      # 10^2 + 20^2 + 30^2 + 40^2

    # power 1 is a distinct column from Sum, not a duplicate of it
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize([Sum(:qty), SumPower(:qty, 1)])))
    @test names(df) == ["time", "qty_sum", "qty_sum1"]
    @test df.qty_sum1 == [100]

    # no identity element: empty input summarizes to missing, not an error
    df = DataFrame(load(Context(0, 9), emptyframe() |>
        summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)])))
    @test df.time == [9]
    @test ismissing(only(df.qty_min))
    @test ismissing(only(df.qty_max))
    @test ismissing(only(df.qty_first))
    @test ismissing(only(df.qty_last))

    # keyed first/last pin down per-group ordering
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize([First(:qty), Last(:qty)]; key = :sym)))
    @test df.sym == ["a", "b"]
    @test df.qty_first == [10, 20]
    @test df.qty_last == [40, 20]

    # key: one row per unique key value, sorted by key
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize([Count(), Sum(:qty)]; key = :sym)))
    @test names(df) == ["time", "sym", "count", "qty_sum"]
    @test df.time == [9, 9]
    @test df.sym == ["a", "b"]
    @test df.count == [3, 1]
    @test df.qty_sum == [80, 20]

    # multi-column key
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarize(Count(); key = [:sym, :qty])))
    @test names(df) == ["time", "sym", "qty", "count"]
    @test df.sym == ["a", "a", "a", "b"]
    @test df.qty == [10, 30, 40, 20]
    @test all(df.count .== 1)

    # empty input: identity row without key, no rows with key
    df = DataFrame(load(Context(0, 9),
        emptyframe() |> summarize([Count(), Sum(:qty)])))
    @test df.time == [9]
    @test df.count == [0]
    @test df.qty_sum == [0]
    @test nrow(load(Context(0, 9),
        emptyframe() |> summarize(Count(); key = :sym))) == 0

    # validation
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path) |> summarize(BadTime()))
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path) |> summarize(Sum(:qty); key = :qty_sum))
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path) |> summarize(Summarizer[]))
end

@testset "summarizecycles" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(path, """
        time,sym,qty
        1,a,10
        2,b,20
        2,a,30
        4,a,40
        """)

    # fresh state per cycle, one row per unique time
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarizecycles([Count(), Sum(:qty)])))
    @test names(df) == ["time", "count", "qty_sum"]
    @test df.time == [1, 2, 4]
    @test df.count == [1, 2, 1]
    @test df.qty_sum == [10, 50, 40]

    # per key within each cycle, sorted by key
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> summarizecycles(Sum(:qty); key = :sym)))
    @test df.time == [1, 2, 2, 4]
    @test df.sym == ["a", "a", "b", "a"]
    @test df.qty_sum == [10, 30, 20, 40]

    # a cycle spanning a chunk boundary is one cycle
    twochunks = CausalPipeline(ctx ->
        [DataFrame(time = [1, 2, 2], qty = [1, 2, 3]),
         DataFrame(time = [2, 3], qty = [4, 5])])
    df = DataFrame(load(Context(0, 9),
        twochunks |> summarizecycles([Count(), Sum(:qty)])))
    @test df.time == [1, 2, 3]
    @test df.count == [1, 3, 1]
    @test df.qty_sum == [1, 9, 5]

    # empty input yields no rows
    @test nrow(load(Context(0, 9), emptyframe() |> summarizecycles(Count()))) == 0
end

@testset "addsummarycolumns" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(path, """
        time,sym,qty
        1,a,10
        2,b,20
        2,a,30
        4,a,40
        """)

    # running values after each row, existing columns kept
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> addsummarycolumns([Count(), Sum(:qty)])))
    @test names(df) == ["time", "sym", "qty", "count", "qty_sum"]
    @test df.count == [1, 2, 3, 4]
    @test df.qty_sum == [10, 30, 60, 100]

    # per-key running values
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> addsummarycolumns(Sum(:qty); key = :sym)))
    @test df.qty_sum == [10, 20, 40, 80]

    # a running min/last never sees an empty group, so never yields missing
    df = DataFrame(load(Context(0, 9),
        readcsv(path) |> addsummarycolumns([Min(:qty), Last(:qty)])))
    @test df.qty_min == [10, 10, 10, 10]
    @test df.qty_last == [10, 20, 30, 40]

    # chunk structure preserved, state carried across the boundary
    twochunks = CausalPipeline(ctx ->
        [DataFrame(time = [1, 2], qty = [1, 2]),
         DataFrame(time = [3, 4], qty = [3, 4])])
    p = twochunks |> addsummarycolumns(Sum(:qty))
    @test length(load(Context(0, 9), p).chunks) == 2
    frames = collect(stream(Context(0, 9), p))
    @test length(frames) == 2
    @test DataFrame(frames[1]).qty_sum == [1, 3]
    @test DataFrame(frames[2]).qty_sum == [6, 10]
    @test context(frames[1]) == Context(0, 3)
    @test context(frames[2]) == Context(3, 9)
    @test DataFrame(load(Context(0, 9), p)).qty_sum == [1, 3, 6, 10]

    # collision with an existing column
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path) |> addsummarycolumns(MinMax(:qty)) |>
            addsummarycolumns(MinMax(:qty)))
end
