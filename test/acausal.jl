using CausalFrames.Acausal

@testset "futurejoin" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])

    @testset "basic" begin
        left = onechunk(time = [1, 2, 3, 5], px = [10.0, 20.0, 30.0, 50.0])
        right = onechunk(time = [0, 2, 4], qty = [1, 2, 3])
        df = DataFrame(load(Context(0, 10), left |> futurejoin(right)))
        @test names(df) == ["time", "px", "qty"]
        @test df.time == [1, 2, 3, 5]
        @test df.px == [10.0, 20.0, 30.0, 50.0]
        # earliest right row at or after each left time
        @test isequal(df.qty, [2, 2, 3, missing])
        @test eltype(df.qty) == Union{Missing,Int}

        # no right row at or after the last left rows -> missing
        early = onechunk(time = [1], qty = [9])
        df = DataFrame(load(Context(0, 10), left |> futurejoin(early)))
        @test isequal(df.qty, [9, missing, missing, missing])
    end

    @testset "ties and equal times" begin
        left = onechunk(time = [2, 2, 3])
        # two right rows at one time: the first in stream order wins
        right = onechunk(time = [2, 2], qty = [1, 2])
        df = DataFrame(load(Context(0, 10), left |> futurejoin(right)))
        @test isequal(df.qty, [1, 1, missing])
    end

    @testset "strict" begin
        left = onechunk(time = [1, 2, 3])
        right = onechunk(time = [2, 3], qty = [1, 2])
        df = DataFrame(load(Context(0, 10),
            left |> futurejoin(right; strict = true)))
        @test isequal(df.qty, [1, 2, missing])
    end

    @testset "keys" begin
        left = onechunk(time = [1, 2, 3, 5], sym = ["a", "b", "a", "b"])
        right = onechunk(time = [2, 4, 6], sym = ["a", "b", "a"],
            qty = [1, 2, 3])
        df = DataFrame(load(Context(0, 10),
            left |> futurejoin(right; key = :sym)))
        @test names(df) == ["time", "sym", "qty"]
        @test df.sym == ["a", "b", "a", "b"]
        # a: 2->1, 6->3; b: 4->2, none
        @test isequal(df.qty, [1, 2, 3, missing])

        # multi-key: both columns must match exactly
        left2 = onechunk(time = [1, 1], sym = ["a", "a"], venue = ["x", "y"])
        right2 = onechunk(time = [2, 3], sym = ["a", "a"],
            venue = ["x", "z"], qty = [1, 2])
        df = DataFrame(
            load(Context(0, 10),
                left2 |> futurejoin(right2; key = [:sym, :venue])),
        )
        @test isequal(df.qty, [1, missing])

        # keys are not prefixed; an explicit empty key is keyless
        df = DataFrame(
            load(Context(0, 10),
                left |> futurejoin(right; key = :sym,
                    leftprefix = "l",
                    rightprefix = "r")),
        )
        @test names(df) == ["time", "sym", "r_qty"]
        p =
            onechunk(time = [1], qty = [1]) |>
            futurejoin(onechunk(time = [2], px = [1.0]); key = Symbol[])
        @test names(DataFrame(load(Context(0, 10), p))) == ["time", "qty", "px"]

        # key validation, each side
        nokey = onechunk(time = [1], qty = [1])
        @test_throws ArgumentError load(Context(0, 10),
            nokey |> futurejoin(right; key = :sym))
        @test_throws ArgumentError load(Context(0, 10),
            left |> futurejoin(nokey; key = :sym))
    end

    @testset "tolerance" begin
        left = onechunk(time = [3, 5], px = [1.0, 2.0])
        right = onechunk(time = [7], qty = [1])
        # inclusive boundary: distance == tolerance matches
        df = DataFrame(load(Context(0, 10),
            left |> futurejoin(right; tolerance = 2)))
        @test isequal(df.qty, [missing, 1])   # 7 - 3 = 4 > 2; 7 - 5 = 2 <= 2

        # the right context is widened forward by the tolerance...
        seen = Ref{Any}(nothing)
        recorder = CausalPipeline() do ctx
            seen[] = ctx
            [DataFrame(time = [ctx.stop - 1], qty = [1])]
        end
        df = DataFrame(load(Context(0, 6),
            left |> futurejoin(recorder; tolerance = 3)))
        @test seen[] == Context(0, 9)
        @test isequal(df.qty, [missing, 1])   # right row at 8; 8-5=3<=3, 8-3=5>3
        # ...and untouched without one
        load(Context(0, 6), left |> futurejoin(recorder))
        @test seen[] == Context(0, 6)

        # tolerance = 0 admits exact-time matches only; never under strict
        exact = onechunk(time = [3], qty = [9])
        df = DataFrame(load(Context(0, 10),
            left |> futurejoin(exact; tolerance = 0)))
        @test isequal(df.qty, [9, missing])
        df = DataFrame(
            load(Context(0, 10),
                left |> futurejoin(exact; tolerance = 0,
                    strict = true)),
        )
        @test isequal(df.qty, [missing, missing])

        # negative tolerance is rejected when the pipeline runs
        p = left |> futurejoin(right; tolerance = -1)
        @test_throws ArgumentError load(Context(0, 10), p)
    end

    @testset "prefixes and duplicates" begin
        left = onechunk(time = [2], sym = ["a"], px = [1.0])
        right = onechunk(time = [3], sym = ["a"], qty = [2])
        df = DataFrame(
            load(Context(0, 10),
                left |> futurejoin(right; key = :sym,
                    leftprefix = :l,
                    rightprefix = "r",
                    righttime = :rt)),
        )
        @test names(df) == ["time", "sym", "l_px", "r_qty", "rt"]
        @test df.rt == [3]

        # righttime is missing where the match is
        df = DataFrame(
            load(Context(0, 10),
                onechunk(time = [9], px = [1.0]) |>
                futurejoin(right; rightprefix = "r", righttime = :rt)),
        )
        @test isequal(df.rt, [missing])

        # duplicate output names: same column both sides, and righttime
        # colliding with a left column
        both = onechunk(time = [1], qty = [1])
        @test_throws ArgumentError load(Context(0, 10),
            both |> futurejoin(both))
        @test_throws ArgumentError load(Context(0, 10),
            left |> futurejoin(right; key = :sym, righttime = :px))

        # construction-time errors
        @test_throws ArgumentError futurejoin(right; key = :time)
        @test_throws ArgumentError futurejoin(right; key = [:sym, :sym])
        @test_throws ArgumentError futurejoin(right; righttime = :time)
        @test_throws ArgumentError futurejoin(right; key = :sym,
            righttime = :sym)
    end

    @testset "self join" begin
        p = onechunk(time = [1, 2, 3], px = [1.0, 2.0, 3.0])
        # strict forward self join gives next-row semantics
        df = DataFrame(
            load(Context(0, 10),
                p |> futurejoin(p; rightprefix = "next",
                    strict = true)),
        )
        @test isequal(df.next_px, [2.0, 3.0, missing])
        df = DataFrame(load(Context(0, 10),
            p |> futurejoin(p; rightprefix = "cur")))
        @test df.cur_px == [1.0, 2.0, 3.0]
        @test_throws ArgumentError load(Context(0, 10), p |> futurejoin(p))
    end

    @testset "multi-chunk" begin
        # interleaved chunk boundaries: matching pauses mid-left-chunk to
        # pull right chunks, and the per-key buffers carry across left chunks
        left = CausalPipeline(ctx -> [DataFrame(time = [1, 4]),
            DataFrame(time = [6, 8])])
        right = CausalPipeline(
            ctx -> [DataFrame(time = [2, 5], qty = [1, 2]),
                DataFrame(time = [7], qty = [3]),
                DataFrame(time = [9], qty = [4])],
        )
        joined = left |> futurejoin(right)
        df = DataFrame(load(Context(0, 10), joined))
        @test isequal(df.qty, [1, 2, 3, 4])
        streamed = reduce(vcat, [DataFrame(f)
                                 for f in stream(Context(0, 10), joined)])
        @test isequal(streamed, df)

        # a right column's eltype may widen between chunks, mid-left-chunk
        wide = CausalPipeline(
            ctx -> [DataFrame(time = [2], qty = [1]),
                DataFrame(time = [4], qty = [2.5])],
        )
        df = DataFrame(
            load(Context(0, 10),
                CausalPipeline(ctx -> [DataFrame(time = [1, 3, 5])]) |>
                futurejoin(wide)),
        )
        @test isequal(df.qty, [1.0, 2.5, missing])
        @test eltype(df.qty) == Union{Missing,Float64}

        # a single early right row satisfies several later left rows before it
        df = DataFrame(
            load(Context(0, 10),
                onechunk(time = [1, 4]) |> futurejoin(onechunk(time = [5],
                    qty = [1]))),
        )
        @test isequal(df.qty, [1, 1])
    end

    @testset "empty streams" begin
        left = onechunk(time = [1, 2], px = [1.0, 2.0])
        # empty right: passthrough, leftprefix still applied
        df = DataFrame(
            load(Context(0, 10),
                left |> futurejoin(emptyframe(); leftprefix = "l",
                    righttime = :rt)),
        )
        @test names(df) == ["time", "l_px"]
        @test df.l_px == [1.0, 2.0]

        # empty left: no frames, and the right side is never pulled
        untouched = CausalPipeline(ctx ->
            (error("right pulled") for _ in 1:1))
        frame = load(Context(0, 10), emptyframe() |> futurejoin(untouched))
        @test nrow(frame) == 0
        @test names(frame) == ["time"]
    end

    @testset "uncurried form" begin
        left = onechunk(time = [1, 2, 3, 5], sym = ["a", "b", "a", "b"])
        right = onechunk(time = [2, 4, 6], sym = ["a", "b", "a"],
            qty = [1, 2, 3])
        # the pipeline-first form matches the |> chain, keywords forwarded
        curried = left |> futurejoin(right; key = :sym, strict = true)
        uncurried = futurejoin(left, right; key = :sym, strict = true)
        @test isequal(DataFrame(load(Context(0, 10), curried)),
            DataFrame(load(Context(0, 10), uncurried)))
    end

    @testset "asofjoin duality" begin
        # futurejoin on time-negated streams is asofjoin on the originals:
        # earliest future match at t mirrors most-recent past match at -t.
        times = [1, 3, 4, 7, 9]
        rtimes = [0, 2, 4, 6, 8]
        left = onechunk(time = times, id = collect(1:5))
        right = onechunk(time = rtimes, qty = collect(10:14))
        asof = DataFrame(load(Context(0, 12), left |> asofjoin(right)))

        negleft = onechunk(time = -reverse(times), id = reverse(1:5))
        negright = onechunk(time = -reverse(rtimes), qty = reverse(10:14))
        fut = DataFrame(
            load(Context(-12, 1),
                negleft |> futurejoin(negright; strict = false)),
        )
        # undo the reversal and compare the matched qty columns
        @test isequal(reverse(fut.qty), asof.qty)
    end
end
