@testset "sortcycles" begin
    ctx = Context(0, 10)
    # A fixed multi-chunk stream, copied per run since consumers own their chunks.
    chunked(dfs...) = CausalPipeline(_ -> [copy(df) for df in dfs])
    ids(p) = DataFrame(load(ctx, p)).id

    src = chunked(
        DataFrame(time = [1, 1, 1, 2, 3, 3], k = ["b", "a", "c", "z", "b", "a"],
            v = [2, 1, 2, 9, 1, 1], id = [1, 2, 3, 4, 5, 6]),
    )

    @testset "keys" begin
        @test ids(src |> sortcycles(:k)) == [2, 1, 3, 4, 6, 5]
        @test ids(src |> sortcycles("k")) == [2, 1, 3, 4, 6, 5]
        # ties keep stream order
        @test ids(src |> sortcycles(:v)) == [2, 1, 3, 4, 5, 6]
        @test ids(src |> sortcycles(:v; rev = true)) == [1, 3, 2, 4, 5, 6]
        @test ids(src |> sortcycles([:v, :k])) == [2, 1, 3, 4, 6, 5]
        @test ids(src |> sortcycles((:v, "k"); rev = true)) == [3, 1, 2, 4, 5, 6]
        @test ids(src |> sortcycles(r -> (-r.v, r.k))) == [1, 3, 2, 4, 6, 5]
        # already in order
        @test ids(src |> sortcycles(:id)) == 1:6

        df = DataFrame(load(ctx, src |> sortcycles(:k)))
        @test names(df) == ["time", "k", "v", "id"]
        @test df.time == [1, 1, 1, 2, 3, 3]
        @test df.k == ["a", "b", "c", "z", "a", "b"]
    end

    @testset "cycles spanning chunks" begin
        # the cycle at 2 spans three chunks, the middle one entirely; the cycle
        # at 3 is still open when the stream ends
        spread = chunked(DataFrame(time = [1, 2, 2], id = [1, 3, 2]),
            DataFrame(time = [2], id = [5]),
            DataFrame(time = [2, 2, 3], id = [4, 0, 7]),
            DataFrame(time = [3], id = [6]))
        df = DataFrame(load(ctx, spread |> sortcycles(:id)))
        @test df.time == [1, 2, 2, 2, 2, 2, 3, 3]
        @test df.id == [1, 0, 2, 3, 4, 5, 6, 7]

        streamed = reduce(vcat, DataFrame.(stream(ctx, spread |> sortcycles(:id))))
        @test streamed == df

        # a chunk closing the held-back cycle without rows of its own at that
        # time, while emitting cycles of its own; then one closing it with a
        # leading row and nothing else to emit
        closing = chunked(DataFrame(time = [1, 1], id = [2, 1]),
            DataFrame(time = [2, 2, 3], id = [4, 3, 5]),
            DataFrame(time = [3, 4], id = [6, 7]))
        @test ids(closing |> sortcycles(:id)) == [1, 2, 3, 4, 5, 6, 7]
        @test ids(closing |> sortcycles(:id; rev = true)) == [2, 1, 4, 3, 6, 5, 7]
        @test ids(closing |> sortcycles(:time)) == [2, 1, 4, 3, 5, 6, 7]
        @test DataFrame(load(ctx, closing |> sortcycles(:id))).time ==
              [1, 1, 2, 2, 3, 3, 4]

        # one cycle for the whole stream
        single = chunked(DataFrame(time = [4, 4], id = [2, 1]),
            DataFrame(time = [4], id = [0]))
        @test ids(single |> sortcycles(:id)) == [0, 1, 2]
        @test ids(single |> sortcycles(r -> r.id; rev = true)) == [2, 1, 0]

        # element types promote across the pieces of a cycle
        drifting = chunked(DataFrame(time = [1, 1], x = [2, 1]),
            DataFrame(time = [1], x = [0.5]))
        x = DataFrame(load(ctx, drifting |> sortcycles(:x))).x
        @test x == [0.5, 1.0, 2.0]
        @test eltype(x) == Float64
    end

    @testset "missing" begin
        m = chunked(DataFrame(time = [1, 1, 1], x = [2, missing, 1]))
        @test isequal(DataFrame(load(ctx, m |> sortcycles(:x))).x, [1, 2, missing])
        @test isequal(DataFrame(load(ctx, m |> sortcycles(:x; rev = true))).x,
            [missing, 2, 1])
    end

    @testset "validation" begin
        @test_throws ArgumentError sortcycles(Symbol[])
        @test_throws ArgumentError sortcycles(1)
        @test_throws ArgumentError sortcycles([:a, 1])
        @test_throws ArgumentError sortcycles(nothing)
        @test_throws ArgumentError sortcycles(src)
        @test_throws ArgumentError load(ctx, src |> sortcycles(:nope))
        @test_throws ArgumentError load(ctx, src |> sortcycles([:k, :nope]))
        reordering = chunked(DataFrame(time = [1], a = [1], b = [2]),
            DataFrame(time = [1], b = [2], a = [1]))
        @test_throws ArgumentError load(ctx, reordering |> sortcycles(:a))
    end

    @testset "split contexts, empty input and uncurried form" begin
        table = DataFrame(time = [0, 0, 3, 3, 3, 5, 7, 7],
            id = [2, 1, 5, 3, 4, 6, 8, 7])
        p = readtable(table) |> sortcycles(:id)
        whole = DataFrame(load(ctx, p))
        @test whole.id == [1, 2, 3, 4, 5, 6, 7, 8]
        @test vcat(DataFrame(load(Context(0, 3), p)),
            DataFrame(load(Context(3, 10), p))) == whole

        @test nrow(load(ctx, emptyframe() |> sortcycles(:x))) == 0

        @test DataFrame(load(ctx, sortcycles(src, :v; rev = true))) ==
              DataFrame(load(ctx, src |> sortcycles(:v; rev = true)))
    end

    @testset "ranking within a timestamp" begin
        films = DataFrame(year = [2001, 2001, 2001, 2002, 2002],
            votes = [10, 30, 20, 5, 5], id = ["c", "a", "b", "e", "d"])
        p =
            readtable(films; time = :year) |>
            sortcycles(r -> (-r.votes, r.id)) |>
            addsummarycolumns(Count(); key = :time)
        df = DataFrame(load(Context(2000, 2010), p))
        @test df.id == ["a", "b", "c", "d", "e"]
        @test df.count == [1, 2, 3, 1, 2]
    end
end
