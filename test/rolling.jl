@testset "addrollingcolumns" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])

    @testset "basic" begin
        p = onechunk(time = [1, 2, 3, 5, 8], x = [10, 20, 30, 40, 50])
        df = DataFrame(load(Context(0, 10),
            p |> addrollingcolumns((w2 = 2,), Sum(:x))))
        @test names(df) == ["time", "x", "w2_x_sum"]
        @test df.time == [1, 2, 3, 5, 8]
        @test df.x == [10, 20, 30, 40, 50]
        # windows: {1} {1,2} {1,2,3} {3,5} {8} — the row itself is included,
        # and rows leaving the look-back are evicted
        @test df.w2_x_sum == [10, 30, 60, 70, 50]
        @test eltype(df.w2_x_sum) == Int
    end

    @testset "multiple windows and prefixes" begin
        p = onechunk(time = [1, 2, 3, 5, 8], x = [10, 20, 30, 40, 50])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w0 = 0, w9 = 9), [Min(:x), Mean(:x)])),
        )
        @test names(df) ==
              ["time", "x", "w0_x_min", "w0_x_mean", "w9_x_min", "w9_x_mean"]
        # zero look-back: only rows at exactly t
        @test isequal(df.w0_x_min, [10, 20, 30, 40, 50])
        @test df.w9_x_min == [10, 10, 10, 10, 10]
        @test df.w9_x_mean == [10.0, 15.0, 20.0, 25.0, 30.0]
        # an empty window would emit missing, so the element types widen
        @test eltype(df.w0_x_min) == Union{Missing,Int}
        @test eltype(df.w9_x_mean) == Union{Missing,Float64}
        # hidden dependencies of Mean are folded but never emitted
        @test !("w9_count" in names(df)) && !("w9_x_sum" in names(df))
    end

    @testset "ties share the whole timestamp" begin
        p = onechunk(time = [1, 2, 2, 3], x = [1, 2, 3, 4])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w0 = 0, w1 = 1), Sum(:x))),
        )
        # every row at time t sees every summarized row at time t
        @test df.w0_x_sum == [1, 5, 5, 4]
        @test df.w1_x_sum == [1, 6, 6, 9]
    end

    @testset "keys" begin
        p = onechunk(time = [1, 1, 2, 3], k = ["a", "b", "a", "c"],
            x = [1, 2, 3, 4])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w5 = 5,), [Sum(:x), Min(:x)]; key = :k)),
        )
        @test df.w5_x_sum == [1, 2, 4, 4]
        # the first row of a key never seen before starts an empty window;
        # for Min the empty value is missing, widening the element type
        @test isequal(df.w5_x_min, [1, 2, 1, 4])
        @test eltype(df.w5_x_min) == Union{Missing,Int}

        # a key with no in-window summarized rows yields the empty values
        src = onechunk(time = [1], k = ["a"], y = [7])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w9 = 9,), [Sum(:y), Min(:y)]; key = :k,
                    from = src)),
        )
        @test isequal(df.w9_y_sum, [7, 0, 7, 0])
        @test isequal(df.w9_y_min, [7, missing, 7, missing])

        # multi-key: both columns must match; empty key collection is keyless
        p2 = onechunk(time = [2, 2], k = ["a", "a"], v = ["x", "y"],
            x = [1, 2])
        df = DataFrame(
            load(Context(0, 10),
                p2 |> addrollingcolumns((w1 = 1,), Sum(:x); key = [:k, :v])),
        )
        @test df.w1_x_sum == [1, 2]
        df = DataFrame(
            load(Context(0, 10),
                p2 |> addrollingcolumns((w1 = 1,), Sum(:x); key = Symbol[])),
        )
        @test df.w1_x_sum == [3, 3]

        # key validation, each side
        nokey = onechunk(time = [1], x = [1])
        @test_throws ArgumentError load(Context(0, 10),
            nokey |> addrollingcolumns((w1 = 1,), Sum(:x); key = :k))
        @test_throws ArgumentError load(Context(0, 10),
            p |> addrollingcolumns((w1 = 1,), Sum(:x); key = :k,
                from = nokey))
    end

    @testset "summarizing a different pipeline" begin
        p = onechunk(time = [1, 2, 3, 5, 8], x = [10, 20, 30, 40, 50])
        src = onechunk(time = [-2, 0, 1, 4], y = [1.0, 2.0, 3.0, 4.0])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w3 = 3,), Sum(:y); from = src)),
        )
        @test names(df) == ["time", "x", "w3_y_sum"]
        # windows: {-2,0,1} {0,1} {0,1} {4} {} — matched by time only
        @test df.w3_y_sum == [6.0, 5.0, 5.0, 4.0, 0.0]
    end

    @testset "context extension" begin
        # the summarized context starts at the earliest windowed start
        seen = Ref{Any}(nothing)
        recorder = CausalPipeline() do ctx
            seen[] = ctx
            [DataFrame(time = [ctx.start], y = [1])]
        end
        p = onechunk(time = [3], x = [1])
        df = DataFrame(
            load(Context(3, 10),
                p |> addrollingcolumns((a = 2, b = 5), Count(); from = recorder)),
        )
        @test seen[] == Context(-2, 10)
        # the pre-window row at -2 is within b (3 - -2 = 5) but not a
        @test df.a_count == [0]
        @test df.b_count == [1]

        # self-summarization runs the pipeline twice: once as is for the
        # rows, once over the widened context for the summaries
        ctxs = []
        selfrec = CausalPipeline() do ctx
            push!(ctxs, ctx)
            [DataFrame(time = [max(ctx.start, 3)], x = [1])]
        end
        load(Context(3, 10), selfrec |> addrollingcolumns((w2 = 2,), Count()))
        @test sort(ctxs; by = c -> c.start) == [Context(1, 10), Context(3, 10)]
    end

    @testset "windows argument forms" begin
        p = onechunk(time = [1, 2, 4], x = [1, 2, 3])
        expected = [1, 3, 5]
        for w in ((w2 = 2,), :w2 => 2, "w2" => 2, [:w2 => 2],
            ["w2" => 2], Dict(:w2 => 2))
            df = DataFrame(load(Context(0, 10),
                p |> addrollingcolumns(w, Sum(:x))))
            @test df.w2_x_sum == expected
        end
    end

    @testset "validation" begin
        p = onechunk(time = [1], x = [1])
        @test_throws ArgumentError addrollingcolumns((;), Sum(:x))
        @test_throws ArgumentError addrollingcolumns([:a => 1, :a => 2],
            Sum(:x))
        @test_throws ArgumentError addrollingcolumns((w = 1,), Summarizer[])
        @test_throws ArgumentError addrollingcolumns((w = 1,), Sum(:x);
            key = :time)
        @test_throws ArgumentError addrollingcolumns((w = 1,), Sum(:x);
            key = [:k, :k])
        # prefixed names collide across windows: a + b_x_sum == a_b + x_sum
        @test_throws ArgumentError addrollingcolumns(
            [:a => 1, :a_b => 2], [Sum(Symbol("b_x")), Sum(:x)])
        # ... and with existing columns, caught when the schema is seen
        taken = onechunk(time = [1], x = [1], w1_x_sum = [9])
        @test_throws ArgumentError load(Context(0, 10),
            taken |> addrollingcolumns((w1 = 1,), Sum(:x)))
        # negative look-back is rejected when the pipeline runs
        q = p |> addrollingcolumns((w = -1,), Sum(:x))
        @test_throws ArgumentError load(Context(0, 10), q)
    end

    @testset "chunk boundaries" begin
        # the summarized stream is pulled on demand, mid-augmented-chunk
        p = onechunk(time = [1, 5, 9], x = [0, 0, 0])
        src = CausalPipeline(
            ctx -> [DataFrame(time = [0, 1], y = [1, 1]),
                DataFrame(time = [2, 4], y = [1, 1]),
                DataFrame(time = [6, 8], y = [1, 1])],
        )
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w2 = 2, w10 = 10), Count(); from = src)),
        )
        @test df.w2_count == [2, 1, 1]
        @test df.w10_count == [2, 4, 6]

        # multi-chunk augmented stream: state carries across chunks, and
        # streaming agrees with loading
        mc = CausalPipeline(
            ctx -> [DataFrame(time = [1, 2], x = [1, 2]),
                DataFrame(time = [3, 4], x = [3, 4])],
        )
        t = addrollingcolumns((w2 = 2,), Sum(:x))
        loaded = DataFrame(load(Context(0, 10), mc |> t))
        @test loaded.w2_x_sum == [1, 3, 6, 9]
        streamed = reduce(vcat, DataFrame.(stream(Context(0, 10), mc |> t)))
        @test streamed == loaded
    end

    @testset "schema widening across summarized chunks" begin
        # Int then Float64: prototypes, buffer, and the half-filled value
        # vectors all widen mid-augmented-chunk
        p = onechunk(time = [1, 2, 3], x = [0, 0, 0])
        src = CausalPipeline(
            ctx -> [DataFrame(time = [1, 2], y = [1, 2]),
                DataFrame(time = [3], y = [2.5])],
        )
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w5 = 5,), Sum(:y); from = src)),
        )
        @test df.w5_y_sum == [1.0, 3.0, 5.5]
        @test eltype(df.w5_y_sum) == Float64
    end

    @testset "empty streams" begin
        p = onechunk(time = [1, 2], x = [1, 2])
        # a summarized stream with no chunks yields the empty values
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w2 = 2,), [Sum(:x), Min(:x)];
                    from = emptyframe())),
        )
        @test names(df) == ["time", "x", "w2_x_sum", "w2_x_min"]
        @test df.w2_x_sum == [0, 0]
        @test all(ismissing, df.w2_x_min)
        # an empty augmented stream stays empty
        f = load(Context(0, 10),
            emptyframe() |> addrollingcolumns((w2 = 2,), Sum(:x);
                from = p))
        @test nrow(DataFrame(f)) == 0
    end

    @testset "buffer compaction" begin
        # enough evicted rows across chunk boundaries to trigger compaction
        chunks = [DataFrame(time = 1:100, x = fill(1, 100)),
            DataFrame(time = 101:200, x = fill(1, 100))]
        p = CausalPipeline(ctx -> chunks)
        df = DataFrame(load(Context(0, 300),
            p |> addrollingcolumns((w1 = 1,), Sum(:x))))
        @test df.w1_x_sum == [1; fill(2, 199)]
    end

    @testset "dates and mixed periods" begin
        t0 = DateTime(2024, 1, 1)
        p = onechunk(time = [t0, t0 + Minute(30), t0 + Minute(62)],
            x = [1, 2, 3])
        seen = Ref{Any}(nothing)
        probe = CausalPipeline() do ctx
            seen[] = ctx
            [DataFrame(time = [t0, t0 + Minute(30), t0 + Minute(62)],
                x = [1, 2, 3])]
        end
        df = DataFrame(
            load(Context(t0, t0 + Hour(2)),
                p |> addrollingcolumns((m5 = Minute(5), h1 = Hour(1)), Sum(:x);
                    from = probe)),
        )
        @test seen[] == Context(t0 - Hour(1), t0 + Hour(2))
        @test df.m5_x_sum == [1, 2, 3]
        @test df.h1_x_sum == [1, 3, 5]
    end

    @testset "dependent and custom summarizers" begin
        p = onechunk(time = [1, 2, 3, 4], x = [1.0, 2.0, 3.0, 4.0])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w1 = 1,), [Std(:x), TestVar(:x)])),
        )
        @test isequal(df.w1_x_std, [NaN, sqrt(0.5), sqrt(0.5), sqrt(0.5)])
        @test isequal(df.w1_x_var, [0.0, 0.25, 0.25, 0.25])
        @test !("w1_count" in names(df)) && !("w1_x_moment_1" in names(df))

        # a multi-valued summarizer emits all its columns, prefixed
        df = DataFrame(load(Context(0, 10),
            p |> addrollingcolumns((w2 = 2,), MinMax(:x))))
        @test names(df) == ["time", "x", "w2_x_min", "w2_x_max"]
        @test isequal(df.w2_x_min, [1.0, 1.0, 1.0, 2.0])
        @test isequal(df.w2_x_max, [1.0, 2.0, 3.0, 4.0])
    end

    @testset "uncurried form" begin
        p = onechunk(time = [1, 2, 3], x = [1, 2, 3], k = ["a", "a", "b"])
        a = DataFrame(
            load(Context(0, 10),
                addrollingcolumns(p, (w2 = 2,), Sum(:x); key = :k)),
        )
        b = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w2 = 2,), Sum(:x); key = :k)),
        )
        @test a == b
    end
end

@testset "rolling fast paths" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])

    # Column-wise agreement, exact for everything but floats, which may
    # carry sliding-sum drift on the running path (and NaN, e.g. a
    # single-row corrected variance, which isapprox rejects).
    function agrees(fast, slow)
        @test names(fast) == names(slow)
        same(x, y) =
            ismissing(x) ? ismissing(y) :
            !ismissing(y) && (isapprox(x, y) || (isnan(x) && isnan(y)))
        for n in names(fast)
            a, b = fast[!, n], slow[!, n]
            if nonmissingtype(eltype(a)) <: AbstractFloat
                @test all(map(same, a, b))
            else
                @test isequal(a, b)
            end
        end
    end

    # A pseudo-random three-chunk stream with tied times and three keys;
    # each fast path is compared against the re-fold fallback, forced by
    # wrapping every summarizer in the structure-hiding Opaque fixture.
    nrows = 300
    times = cumsum(lcgsequence(1, nrows, 2))          # 0/1 steps: many ties
    xs = map(v -> v - 3, lcgsequence(2, nrows, 7))    # -3:3, with zeros
    ys = map(v -> v - 2, lcgsequence(3, nrows, 5))    # -2:2
    ks = map(v -> ("a", "b", "c")[v+1], lcgsequence(4, nrows, 3))
    ranges = [1:100, 101:220, 221:300]
    mkdata(x) = CausalPipeline(
        ctx ->
            [DataFrame(time = times[r], k = ks[r], x = x[r], y = ys[r])
                for r in ranges],
    )
    intdata = mkdata(xs)
    floatdata = mkdata(Float64.(xs) ./ 4)

    windows = (w0 = 0, w3 = 3, w20 = 20)
    rolled(p, ss; kwargs...) = DataFrame(load(Context(0, 1000),
        p |> addrollingcolumns(windows, ss; kwargs...)))

    groupset = [Count(), Sum(:x), Mean(:x), Variance(:x), Correlation(:x, :y)]
    monoidset = [Min(:x), Max(:x), First(:x), Last(:x), Product(:x)]
    mixedset = [Sum(:x), Min(:x), MinMax(:y)]     # group ⊂ monoid: tree path
    plainset = [Sum(:x), TestVar(:x)]             # plain present: re-fold

    @testset "differential against the re-fold oracle" begin
        for p in (intdata, floatdata), ss in (groupset, monoidset, mixedset,
                plainset)

            agrees(rolled(p, ss), rolled(p, map(Opaque, ss)))
            agrees(rolled(p, ss; key = :k),
                rolled(p, map(Opaque, ss); key = :k))
        end
    end

    @testset "streaming agrees with loading" begin
        for ss in (groupset, monoidset)
            t = addrollingcolumns(windows, ss; key = :k)
            loaded = DataFrame(load(Context(0, 1000), intdata |> t))
            streamed = reduce(vcat,
                DataFrame.(stream(Context(0, 1000),
                    intdata |> t)))
            @test isequal(streamed, loaded)
        end
    end

    @testset "running path empty windows emit the empty values" begin
        # a key that empties out is deleted from its running group, so the
        # empty window emits emptyvalue — Mean gives missing, never 0/0
        p = onechunk(time = [1, 2, 5], k = ["a", "b", "a"], x = [0, 0, 0])
        src = onechunk(time = [1], k = ["a"], y = [7])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w2 = 2,), [Sum(:y), Mean(:y)];
                    key = :k, from = src)),
        )
        @test isequal(df.w2_y_sum, [7, 0, 0])
        @test isequal(df.w2_y_mean, [7.0, missing, missing])
        @test eltype(df.w2_y_mean) == Union{Missing,Float64}
    end

    @testset "missing widens away the running inverse" begin
        # chunk two lets missing into the sum accumulator, demoting the
        # running mode to the tree mid-stream: the poisoned row damages its
        # own windows but recovers once it expires, because a tree query
        # never combines an expired leaf
        p = onechunk(time = [1, 2, 3, 10], x = [0, 0, 0, 0])
        src = CausalPipeline(
            ctx ->
                [DataFrame(time = [1, 2], y = [1, 2]),
                    DataFrame(time = [3, 10], y = [missing, 5])],
        )
        df = DataFrame(
            load(Context(0, 20),
                p |> addrollingcolumns((w2 = 2,), [Sum(:y), Mean(:y)];
                    from = src)),
        )
        @test eltype(df.w2_y_sum) == Union{Missing,Int}
        @test df.w2_y_sum[2] == 3
        @test ismissing(df.w2_y_sum[3]) && ismissing(df.w2_y_mean[3])
        @test df.w2_y_sum[4] == 5 && df.w2_y_mean[4] == 5.0
    end

    @testset "nonfinite rows recover in running mode" begin
        # a float accumulator counts NaN and ±Inf inputs instead of folding
        # them in, so the running path stays running (no demotion) and the
        # window recovers exactly once the nonfinite row expires — a naive
        # running sum would emit NaN forever
        for bad in (NaN, Inf)
            p = onechunk(time = [1, 2, 3, 10], y = [1.0, 2.0, bad, 5.0])
            df = DataFrame(
                load(Context(0, 20),
                    p |> addrollingcolumns((w2 = 2,), [Sum(:y), Mean(:y)])),
            )
            @test eltype(df.w2_y_sum) == Float64
            @test df.w2_y_sum[1:2] == [1.0, 3.0]
            @test isequal(df.w2_y_sum[3], bad + 3.0)
            @test df.w2_y_sum[4] == 5.0 && df.w2_y_mean[4] == 5.0
        end
    end

    @testset "compensated sliding accuracy" begin
        # once the large value leaves the window, the compensation term is
        # all that remains — the naive running sum would emit 0.0, which not
        # even the oracle's isapprox would accept
        p = onechunk(time = [1, 2, 3], x = [1e16, 1.0, 1.0])
        df = DataFrame(load(Context(0, 10),
            p |> addrollingcolumns((w1 = 1,), Sum(:x))))
        @test df.w1_x_sum[3] == 2.0

        # the tree path combines the same compensated states
        p = onechunk(time = [1, 2, 3, 4], x = [1.0, 1e100, 1.0, -1e100])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w9 = 9,), [Sum(:x), Min(:x)])),
        )
        @test df.w9_x_sum[4] == 2.0
    end

    @testset "tree path order sensitivity" begin
        # First/Last through ties: every row at time t sees all rows tied
        # at t, in stream order
        p = onechunk(time = [1, 2, 2, 3], x = [1, 2, 3, 4])
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w1 = 1,), [First(:x), Last(:x)])),
        )
        @test isequal(df.w1_x_first, [1, 1, 1, 2])
        @test isequal(df.w1_x_last, [1, 3, 3, 4])
    end

    @testset "fast paths widen like the re-fold path" begin
        # Int then Float64 across summarized chunks, on both fast paths
        p = onechunk(time = [1, 2, 3], x = [0, 0, 0])
        src = CausalPipeline(
            ctx -> [DataFrame(time = [1, 2], y = [1, 2]),
                DataFrame(time = [3], y = [2.5])],
        )
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w5 = 5,), [Sum(:y), Mean(:y)];
                    from = src)),
        )
        @test df.w5_y_sum == [1.0, 3.0, 5.5]
        @test eltype(df.w5_y_sum) == Float64
        df = DataFrame(
            load(Context(0, 10),
                p |> addrollingcolumns((w5 = 5,), [Sum(:y), Min(:y)];
                    from = src)),
        )
        @test df.w5_y_sum == [1.0, 3.0, 5.5]
        @test isequal(df.w5_y_min, [1.0, 1.0, 1.0])
    end

    @testset "fast paths over dates and mixed periods" begin
        t0 = DateTime(2024, 1, 1)
        p = onechunk(time = [t0, t0 + Minute(30), t0 + Minute(62)],
            x = [1, 2, 3])
        for (ss, sumcol, expect) in
            (([Sum(:x), Mean(:x)], :h1_x_sum, [1, 3, 5]),
            ([Min(:x), Sum(:x)], :h1_x_min, [1, 1, 2]))
            df = DataFrame(
                load(Context(t0, t0 + Hour(2)),
                    p |> addrollingcolumns((m5 = Minute(5), h1 = Hour(1)), ss)),
            )
            @test df[!, sumcol] == expect
        end
    end
end
