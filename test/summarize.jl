@testset "summarize" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(
        path,
        """
time,sym,qty
1,a,10
2,b,20
2,a,30
4,a,40
""",
    )
    tt = Dict(:time => Int, :qty => Int)

    # no key: one row at ctx.stop, input columns dropped
    frame =
        load(Context(0, 9), readcsv(path; types = tt) |> summarize([Count(), Sum(:qty)]))
    df = DataFrame(frame)
    @test names(df) == ["time", "count", "qty_sum"]
    @test df.time == [9]
    @test df.count == [4]
    @test df.qty_sum == [100]

    # a single summarizer (not a collection) works too
    df = DataFrame(load(Context(0, 9), readcsv(path; types = tt) |> summarize(Count())))
    @test df.count == [4]

    # multi-valued summarizer
    df =
        DataFrame(load(Context(0, 9), readcsv(path; types = tt) |> summarize(MinMax(:qty))))
    @test names(df) == ["time", "qty_min", "qty_max"]
    @test df.qty_min == [10]
    @test df.qty_max == [40]

    # duplicate configurations dedup to one column
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Sum(:qty), Sum(:qty)])),
    )
    @test names(df) == ["time", "qty_sum"]

    # min, max, first, last
    df = DataFrame(
        load(
            Context(0, 9),
            readcsv(path; types = tt) |>
            summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)]),
        ),
    )
    @test names(df) == ["time", "qty_min", "qty_max", "qty_first", "qty_last"]
    @test df.qty_min == [10]
    @test df.qty_max == [40]
    @test df.qty_first == [10]
    @test df.qty_last == [40]

    # sum of elements to the Nth power; :x_sumpower_2 never dedups against
    # :x_sum
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Sum(:qty), SumPower(:qty, 2)])),
    )
    @test names(df) == ["time", "qty_sum", "qty_sumpower_2"]
    @test df.qty_sum == [100]
    @test df.qty_sumpower_2 == [3000]      # 10^2 + 20^2 + 30^2 + 40^2

    # power 1 is a distinct column from Sum, not a duplicate of it
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Sum(:qty), SumPower(:qty, 1)])),
    )
    @test names(df) == ["time", "qty_sum", "qty_sumpower_1"]
    @test df.qty_sumpower_1 == [100]

    # no identity element: empty input summarizes to missing, not an error
    df = DataFrame(
        load(
            Context(0, 9),
            emptyframe() |>
            summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)]),
        ),
    )
    @test df.time == [9]
    @test ismissing(only(df.qty_min))
    @test ismissing(only(df.qty_max))
    @test ismissing(only(df.qty_first))
    @test ismissing(only(df.qty_last))

    # keyed first/last pin down per-group ordering
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([First(:qty), Last(:qty)]; key = :sym)),
    )
    @test df.sym == ["a", "b"]
    @test df.qty_first == [10, 20]
    @test df.qty_last == [40, 20]

    # key: one row per unique key value, sorted by key
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Count(), Sum(:qty)]; key = :sym)),
    )
    @test names(df) == ["time", "sym", "count", "qty_sum"]
    @test df.time == [9, 9]
    @test df.sym == ["a", "b"]
    @test df.count == [3, 1]
    @test df.qty_sum == [80, 20]

    # multi-column key
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize(Count(); key = [:sym, :qty])),
    )
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
        readcsv(path; types = tt) |> summarize(BadTime()))
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path; types = tt) |> summarize(Sum(:qty); key = :qty_sum))
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path; types = tt) |> summarize(Summarizer[]))
end

@testset "dependent summarizers" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(
        path,
        """
time,sym,qty
1,a,10
2,b,20
2,a,30
4,a,40
""",
    )
    tt = Dict(:time => Int, :qty => Int)

    # dependencies are folded but hidden: only the requested column is emitted
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize(Moment(:qty, 2))),
    )
    @test names(df) == ["time", "qty_moment_2"]
    @test df.qty_moment_2 == [750.0]     # 3000 / 4

    # Moment(column, 1) is the mean
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize(Moment(:qty, 1))),
    )
    @test names(df) == ["time", "qty_moment_1"]
    @test df.qty_moment_1 == [25.0]

    # a dependency that is also requested appears in the output, in request
    # order, sharing one folded state with the dependent
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Count(), Moment(:qty, 2)])),
    )
    @test names(df) == ["time", "count", "qty_moment_2"]
    @test df.count == [4]
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Moment(:qty, 2), SumPower(:qty, 2)])),
    )
    @test names(df) == ["time", "qty_moment_2", "qty_sumpower_2"]
    @test df.qty_moment_2 == [750.0]
    @test df.qty_sumpower_2 == [3000]

    # duplicate requests dedup to one column
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([Moment(:qty, 2), Moment(:qty, 2)])),
    )
    @test names(df) == ["time", "qty_moment_2"]

    # per-key moments
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize(Moment(:qty, 1); key = :sym)),
    )
    @test df.sym == ["a", "b"]
    @test df.qty_moment_1 == [80 / 3, 20.0]

    # dependencies of dependencies expand transitively
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize(TestVar(:qty))),
    )
    @test names(df) == ["time", "qty_var"]
    @test df.qty_var == [125.0]          # 750 - 25^2
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarize([TestVar(:qty), Moment(:qty, 1)])),
    )
    @test names(df) == ["time", "qty_var", "qty_moment_1"]
    @test df.qty_moment_1 == [25.0]

    # no identity element: the moment of no rows is missing
    df = DataFrame(load(Context(0, 9),
        emptyframe() |> summarize(Moment(:qty, 2))))
    @test ismissing(only(df.qty_moment_2))

    # a dependency cycle is an error
    @test_throws ArgumentError load(Context(0, 9),
        readcsv(path; types = tt) |> summarize(Loopy()))

    # a hidden dependency never reaches the output, so its name may coincide
    # with a key column
    withcount = CausalPipeline(
        ctx ->
            [DataFrame(time = [1, 2], count = ["a", "b"], qty = [10, 20])],
    )
    df = DataFrame(
        load(Context(0, 9),
            withcount |> summarize(Moment(:qty, 1); key = :count)),
    )
    @test names(df) == ["time", "count", "qty_moment_1"]
    @test df.qty_moment_1 == [10.0, 20.0]

    # per-cycle moments, fresh state per cycle
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarizecycles(Moment(:qty, 1))),
    )
    @test names(df) == ["time", "qty_moment_1"]
    @test df.time == [1, 2, 4]
    @test df.qty_moment_1 == [10.0, 25.0, 40.0]

    # running moments; hidden dependency columns are not appended, so they
    # may also coincide with an existing input column
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> addsummarycolumns(Moment(:qty, 1))),
    )
    @test names(df) == ["time", "sym", "qty", "qty_moment_1"]
    @test df.qty_moment_1 == [10.0, 15.0, 20.0, 25.0]
    hascount = CausalPipeline(ctx ->
        [DataFrame(time = [1, 2], count = [7, 7], qty = [10, 20])])
    df = DataFrame(load(Context(0, 9),
        hascount |> addsummarycolumns(Moment(:qty, 1))))
    @test names(df) == ["time", "count", "qty", "qty_moment_1"]
    @test df.qty_moment_1 == [10.0, 15.0]
end

@testset "summarizecycles" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(
        path,
        """
time,sym,qty
1,a,10
2,b,20
2,a,30
4,a,40
""",
    )
    tt = Dict(:time => Int, :qty => Int)

    # fresh state per cycle, one row per unique time
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarizecycles([Count(), Sum(:qty)])),
    )
    @test names(df) == ["time", "count", "qty_sum"]
    @test df.time == [1, 2, 4]
    @test df.count == [1, 2, 1]
    @test df.qty_sum == [10, 50, 40]

    # per key within each cycle, sorted by key
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> summarizecycles(Sum(:qty); key = :sym)),
    )
    @test df.time == [1, 2, 2, 4]
    @test df.sym == ["a", "a", "b", "a"]
    @test df.qty_sum == [10, 30, 20, 40]

    # a cycle spanning a chunk boundary is one cycle
    twochunks = CausalPipeline(
        ctx ->
            [DataFrame(time = [1, 2, 2], qty = [1, 2, 3]),
                DataFrame(time = [2, 3], qty = [4, 5])],
    )
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
    write(
        path,
        """
time,sym,qty
1,a,10
2,b,20
2,a,30
4,a,40
""",
    )
    tt = Dict(:time => Int, :qty => Int)

    # running values after each row, existing columns kept
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> addsummarycolumns([Count(), Sum(:qty)])),
    )
    @test names(df) == ["time", "sym", "qty", "count", "qty_sum"]
    @test df.count == [1, 2, 3, 4]
    @test df.qty_sum == [10, 30, 60, 100]

    # per-key running values
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> addsummarycolumns(Sum(:qty); key = :sym)),
    )
    @test df.qty_sum == [10, 20, 40, 80]

    # a running min/last never sees an empty group, so never yields missing
    df = DataFrame(
        load(Context(0, 9),
            readcsv(path; types = tt) |> addsummarycolumns([Min(:qty), Last(:qty)])),
    )
    @test df.qty_min == [10, 10, 10, 10]
    @test df.qty_last == [10, 20, 30, 40]

    # chunk structure preserved, state carried across the boundary
    twochunks = CausalPipeline(
        ctx ->
            [DataFrame(time = [1, 2], qty = [1, 2]),
                DataFrame(time = [3, 4], qty = [3, 4])],
    )
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
        readcsv(path; types = tt) |> addsummarycolumns(MinMax(:qty)) |>
        addsummarycolumns(MinMax(:qty)))
end

@testset "uncurried summarization forms" begin
    path = joinpath(mktempdir(), "trades.csv")
    write(
        path,
        """
time,sym,qty
1,a,10
2,b,20
2,a,30
4,a,40
""",
    )
    tt = Dict(:time => Int, :qty => Int)
    src = readcsv(path; types = tt)

    # the pipeline-first forms match the |> chain, including the key keyword
    for (curried, uncurried) in (
        (src |> summarize([Count(), Sum(:qty)]),
            summarize(src, [Count(), Sum(:qty)])),
        (src |> summarize(Sum(:qty); key = :sym),
            summarize(src, Sum(:qty); key = :sym)),
        (src |> summarizecycles(Sum(:qty)),
            summarizecycles(src, Sum(:qty))),
        (src |> addsummarycolumns(Sum(:qty); key = :sym),
            addsummarycolumns(src, Sum(:qty); key = :sym)))
        @test DataFrame(load(Context(0, 9), curried)) ==
              DataFrame(load(Context(0, 9), uncurried))
    end
end
