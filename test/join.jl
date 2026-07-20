@testset "asofjoin" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])

    @testset "basic" begin
        left = onechunk(time = [1, 2, 3, 5], px = [10.0, 20.0, 30.0, 50.0])
        right = onechunk(time = [0, 2, 4], qty = [1, 2, 3])
        df = DataFrame(load(Context(0, 10), left |> asofjoin(right)))
        @test names(df) == ["time", "px", "qty"]
        @test df.time == [1, 2, 3, 5]
        @test df.px == [10.0, 20.0, 30.0, 50.0]
        @test isequal(df.qty, [1, 2, 2, 3])
        @test eltype(df.qty) == Union{Missing,Int}

        # no right row at or before the first left rows -> missing
        late = onechunk(time = [4], qty = [9])
        df = DataFrame(load(Context(0, 10), left |> asofjoin(late)))
        @test isequal(df.qty, [missing, missing, missing, 9])
    end

    @testset "ties and equal times" begin
        left = onechunk(time = [2, 2, 3])
        # two right rows at one time: the later in stream order wins
        right = onechunk(time = [2, 2], qty = [1, 2])
        df = DataFrame(load(Context(0, 10), left |> asofjoin(right)))
        @test isequal(df.qty, [2, 2, 2])
    end

    @testset "strict" begin
        left = onechunk(time = [1, 2, 3])
        right = onechunk(time = [1, 2], qty = [1, 2])
        df = DataFrame(load(Context(0, 10),
            left |> asofjoin(right; strict = true)))
        @test isequal(df.qty, [missing, 1, 2])
    end

    @testset "keys" begin
        left = onechunk(time = [1, 2, 3, 5], sym = ["a", "b", "a", "b"])
        right = onechunk(time = [0, 2, 4], sym = ["a", "b", "b"],
            qty = [1, 2, 3])
        df = DataFrame(load(Context(0, 10),
            left |> asofjoin(right; key = :sym)))
        @test names(df) == ["time", "sym", "qty"]
        @test df.sym == ["a", "b", "a", "b"]
        @test isequal(df.qty, [1, 2, 1, 3])

        # multi-key: both columns must match exactly
        left2 = onechunk(time = [3, 3], sym = ["a", "a"], venue = ["x", "y"])
        right2 = onechunk(time = [1, 2], sym = ["a", "a"],
            venue = ["x", "z"], qty = [1, 2])
        df = DataFrame(load(Context(0, 10),
            left2 |> asofjoin(right2; key = [:sym, :venue])))
        @test isequal(df.qty, [1, missing])

        # keys are not prefixed; an explicit empty key is keyless
        df = DataFrame(
            load(Context(0, 10),
                left |> asofjoin(right; key = :sym,
                    leftprefix = "l",
                    rightprefix = "r")),
        )
        @test names(df) == ["time", "sym", "r_qty"]
        p =
            onechunk(time = [1], qty = [1]) |>
            asofjoin(onechunk(time = [0], px = [1.0]); key = Symbol[])
        @test names(DataFrame(load(Context(0, 10), p))) == ["time", "qty", "px"]

        # key validation, each side
        nokey = onechunk(time = [1], qty = [1])
        @test_throws ArgumentError load(Context(0, 10),
            nokey |> asofjoin(right; key = :sym))
        @test_throws ArgumentError load(Context(0, 10),
            left |> asofjoin(nokey; key = :sym))
    end

    @testset "tolerance" begin
        left = onechunk(time = [3, 5], px = [1.0, 2.0])
        right = onechunk(time = [1], qty = [1])
        # inclusive boundary: distance == tolerance matches
        df = DataFrame(load(Context(0, 10),
            left |> asofjoin(right; tolerance = 2)))
        @test isequal(df.qty, [1, missing])   # 5 - 1 = 4 > 2: gone stale

        # the right context is widened by the tolerance...
        seen = Ref{Any}(nothing)
        recorder = CausalPipeline() do ctx
            seen[] = ctx
            [DataFrame(time = [ctx.start], qty = [1])]
        end
        df = DataFrame(load(Context(3, 10),
            left |> asofjoin(recorder; tolerance = 2)))
        @test seen[] == Context(1, 10)
        @test isequal(df.qty, [1, missing])   # pre-window right row matched
        # ...and untouched without one
        load(Context(3, 10), left |> asofjoin(recorder))
        @test seen[] == Context(3, 10)

        # tolerance = 0 admits exact-time matches only; never under strict
        exact = onechunk(time = [3], qty = [9])
        df = DataFrame(load(Context(0, 10),
            left |> asofjoin(exact; tolerance = 0)))
        @test isequal(df.qty, [9, missing])
        df = DataFrame(
            load(Context(0, 10),
                left |> asofjoin(exact; tolerance = 0,
                    strict = true)),
        )
        @test isequal(df.qty, [missing, missing])

        # negative tolerance is rejected when the pipeline runs
        p = left |> asofjoin(right; tolerance = -1)
        @test_throws ArgumentError load(Context(0, 10), p)
    end

    @testset "prefixes and duplicates" begin
        left = onechunk(time = [2], sym = ["a"], px = [1.0])
        right = onechunk(time = [1], sym = ["a"], qty = [2])
        df = DataFrame(
            load(Context(0, 10),
                left |> asofjoin(right; key = :sym,
                    leftprefix = :l,
                    rightprefix = "r",
                    righttime = :rt)),
        )
        @test names(df) == ["time", "sym", "l_px", "r_qty", "rt"]
        @test df.rt == [1]

        # righttime is missing where the match is
        df = DataFrame(
            load(Context(0, 10),
                onechunk(time = [0], px = [1.0]) |>
                asofjoin(right; rightprefix = "r", righttime = :rt)),
        )
        @test isequal(df.rt, [missing])

        # duplicate output names: same column both sides, and righttime
        # colliding with a left column
        both = onechunk(time = [1], qty = [1])
        @test_throws ArgumentError load(Context(0, 10),
            both |> asofjoin(both))
        @test_throws ArgumentError load(Context(0, 10),
            left |> asofjoin(right; key = :sym, righttime = :px))

        # construction-time errors
        @test_throws ArgumentError asofjoin(right; key = :time)
        @test_throws ArgumentError asofjoin(right; key = [:sym, :sym])
        @test_throws ArgumentError asofjoin(right; righttime = :time)
        @test_throws ArgumentError asofjoin(right; key = :sym,
            righttime = :sym)
    end

    @testset "self join" begin
        p = onechunk(time = [1, 2, 3], px = [1.0, 2.0, 3.0])
        df = DataFrame(
            load(Context(0, 10),
                p |> asofjoin(p; rightprefix = "prev",
                    strict = true)),
        )
        @test isequal(df.prev_px, [missing, 1.0, 2.0])
        df = DataFrame(load(Context(0, 10),
            p |> asofjoin(p; rightprefix = "cur")))
        @test df.cur_px == [1.0, 2.0, 3.0]
        @test_throws ArgumentError load(Context(0, 10), p |> asofjoin(p))
    end

    @testset "multi-chunk" begin
        # interleaved chunk boundaries: matching pauses mid-left-chunk to
        # pull right chunks, and the store carries across left chunks
        left = CausalPipeline(ctx -> [DataFrame(time = [1, 4]),
            DataFrame(time = [6, 8])])
        right = CausalPipeline(
            ctx -> [DataFrame(time = [0, 3], qty = [1, 2]),
                DataFrame(time = [5], qty = [3]),
                DataFrame(time = [7], qty = [4])],
        )
        joined = left |> asofjoin(right)
        df = DataFrame(load(Context(0, 10), joined))
        @test isequal(df.qty, [1, 2, 3, 4])
        streamed = reduce(vcat, [DataFrame(f)
                                 for f in stream(Context(0, 10), joined)])
        @test isequal(streamed, df)

        # a right column's eltype may widen between chunks, mid-left-chunk
        wide = CausalPipeline(
            ctx -> [DataFrame(time = [0], qty = [1]),
                DataFrame(time = [2], qty = [2.5])],
        )
        df = DataFrame(
            load(Context(0, 10),
                CausalPipeline(ctx -> [DataFrame(time = [1, 3, 4])]) |>
                asofjoin(wide)),
        )
        @test isequal(df.qty, [1.0, 2.5, 2.5])
        @test eltype(df.qty) == Union{Missing,Float64}

        # right exhausted mid-window: later left rows match from the store
        df = DataFrame(
            load(Context(0, 10),
                onechunk(time = [1, 9]) |> asofjoin(onechunk(time = [0],
                    qty = [1]))),
        )
        @test isequal(df.qty, [1, 1])
    end

    @testset "empty streams" begin
        left = onechunk(time = [1, 2], px = [1.0, 2.0])
        # empty right: passthrough, leftprefix still applied
        df = DataFrame(
            load(Context(0, 10),
                left |> asofjoin(emptyframe(); leftprefix = "l",
                    righttime = :rt)),
        )
        @test names(df) == ["time", "l_px"]
        @test df.l_px == [1.0, 2.0]

        # empty left: no frames, and the right side is never pulled
        untouched = CausalPipeline(ctx ->
            (error("right pulled") for _ in 1:1))
        frame = load(Context(0, 10), emptyframe() |> asofjoin(untouched))
        @test nrow(frame) == 0
        @test names(frame) == ["time"]
    end

    @testset "uncurried form" begin
        left = onechunk(time = [1, 2, 3, 5], sym = ["a", "b", "a", "b"])
        right = onechunk(time = [0, 2, 4], sym = ["a", "b", "b"],
            qty = [1, 2, 3])
        # the pipeline-first form matches the |> chain, keywords forwarded
        curried = left |> asofjoin(right; key = :sym, strict = true)
        uncurried = asofjoin(left, right; key = :sym, strict = true)
        @test isequal(DataFrame(load(Context(0, 10), curried)),
            DataFrame(load(Context(0, 10), uncurried)))
    end
end
