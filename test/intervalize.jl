@testset "intervalize" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])
    multichunk(chunks) = CausalPipeline(ctx -> chunks)

    @testset "keyless regular grid" begin
        # clock(5) over [0, 15) gives boundaries 0, 5, 10; complete intervals
        # [0, 5) -> 5 and [5, 10) -> 10; trailing [10, 15) dropped.
        p = onechunk(time = [2, 7, 7, 12], x = [10, 20, 30, 40])
        df = DataFrame(
            load(Context(0, 15),
                p |> intervalize(clock(5), [Count(), Sum(:x), Mean(:x)])),
        )
        @test names(df) == ["time", "count", "x_sum", "x_mean"]
        @test df.time == [5, 10]
        @test df.count == [1, 2]
        @test df.x_sum == [10, 50]
        @test df.x_mean == [10.0, 25.0]
        # the grid can hold empty intervals, so element types widen to admit
        # the empty values
        @test eltype(df.x_mean) == Union{Missing,Float64}

        # a single summarizer (not a collection) works too
        df = DataFrame(load(Context(0, 15), p |> intervalize(clock(5), Count())))
        @test df.count == [1, 2]
    end

    @testset "empty intervals emit identity/missing values" begin
        # only [0, 5) holds a row; [5, 10) and [10, 15) are empty
        p = onechunk(time = [2], x = [10])
        df = DataFrame(
            load(Context(0, 20),
                p |> intervalize(clock(5), [Count(), Sum(:x), Min(:x), Mean(:x)])),
        )
        @test df.time == [5, 10, 15]
        @test df.count == [1, 0, 0]                 # Count identity is 0
        @test df.x_sum == [10, 0, 0]                # Sum identity is 0
        @test isequal(df.x_min, [10, missing, missing])
        @test isequal(df.x_mean, [10.0, missing, missing])
        @test eltype(df.x_min) == Union{Missing,Int}
    end

    @testset "timestamp is the interval end; membership is half-open" begin
        # a row exactly at boundary 5 belongs to [5, 10), not [0, 5)
        p = onechunk(time = [5], x = [99])
        df = DataFrame(load(Context(0, 15), p |> intervalize(clock(5), Sum(:x))))
        @test df.time == [5, 10]
        @test df.x_sum == [0, 99]
    end

    @testset "rows before the first boundary are dropped" begin
        # clock starting at 3 leaves [0, 3) with no interval
        clk = onechunk(time = [3, 6, 9])
        p = onechunk(time = [1, 2, 4, 7], x = [1, 2, 3, 4])
        df = DataFrame(load(Context(0, 10), p |> intervalize(clk, [Count(), Sum(:x)])))
        # intervals [3, 6) -> 6 (t = 4) and [6, 9) -> 9 (t = 7); t = 1, 2 dropped
        @test df.time == [6, 9]
        @test df.count == [1, 1]
        @test df.x_sum == [3, 4]
    end

    @testset "closelast emits the trailing partial at stop" begin
        p = onechunk(time = [2, 7, 12], x = [10, 20, 30])
        # boundaries 0, 5, 10, 15 over [0, 17); trailing [15, 17)
        without = DataFrame(load(Context(0, 17),
            p |> intervalize(clock(5), Count())))
        @test without.time == [5, 10, 15]
        with = DataFrame(
            load(Context(0, 17),
                p |> intervalize(clock(5), Count(); closelast = true)),
        )
        @test with.time == [5, 10, 15, 17]
        @test with.count == [1, 1, 1, 0]           # trailing interval is empty
    end

    @testset "keyed is sparse: one row per present key, sorted" begin
        p = onechunk(time = [2, 7, 7, 12], k = ["a", "b", "a", "c"],
            x = [10, 20, 30, 40])
        df = DataFrame(
            load(Context(0, 15),
                p |> intervalize(clock(5), [Count(), Sum(:x)]; key = :k)),
        )
        @test names(df) == ["time", "k", "count", "x_sum"]
        # [0, 5) -> 5: {a}; [5, 10) -> 10: {a, b} sorted; t = 12 (c) trailing, dropped
        @test df.time == [5, 10, 10]
        @test df.k == ["a", "a", "b"]
        @test df.count == [1, 1, 1]
        @test df.x_sum == [10, 30, 20]

        # an interval with no rows for any key emits nothing (no empty grid)
        sparse = onechunk(time = [2], k = ["a"], x = [10])
        dfs = DataFrame(
            load(Context(0, 15),
                sparse |> intervalize(clock(5), Count(); key = :k)),
        )
        @test dfs.time == [5]
        @test dfs.k == ["a"]

        # closelast keyed: the trailing interval's present keys at stop
        dfc = DataFrame(
            load(Context(0, 15),
                p |> intervalize(clock(5), Count(); key = :k, closelast = true)),
        )
        @test dfc.time == [5, 10, 10, 15]
        @test dfc.k == ["a", "a", "b", "c"]        # trailing {c} at stop
    end

    @testset "declared keyset is dense" begin
        p = onechunk(time = [2, 7, 7, 12], k = ["a", "b", "a", "c"],
            x = [10, 20, 30, 40])
        run(ss; kwargs...) = DataFrame(
            load(Context(0, 15),
                p |> intervalize(clock(5), ss; key = :k, kwargs...)),
        )
        # every interval emits every declared key, in declared order, empty
        # values included; t = 12 (c) is trailing and dropped
        df = run([Count(), Sum(:x), Min(:x)]; keyset = ["c", "a", "b"])
        @test names(df) == ["time", "k", "count", "x_sum", "x_min"]
        @test df.time == [5, 5, 5, 10, 10, 10]
        @test df.k == ["c", "a", "b", "c", "a", "b"]
        @test df.count == [0, 1, 0, 0, 1, 1]
        @test df.x_sum == [0, 10, 0, 0, 30, 20]
        @test isequal(df.x_min, [missing, 10, missing, missing, 30, 20])
        @test eltype(df.x_min) == Union{Missing,Int}

        # closelast closes the trailing interval over every key too
        df = run(Count(); keyset = ["a", "b", "c"], closelast = true)
        @test df.time == [5, 5, 5, 10, 10, 10, 15, 15, 15]
        @test df.count == [1, 0, 0, 1, 1, 0, 0, 0, 1]

        # an interval spanning chunks, a widening inside it, and an empty
        # interval closed by a row that lands past the last boundary
        q = multichunk([DataFrame(time = [1, 6], k = ["a", "b"], x = [1, 2]),
            DataFrame(time = [8, 16], k = ["b", "a"], x = [2.5, 4.0])])
        df = DataFrame(
            load(Context(0, 20),
                q |> intervalize(clock(5), [Count(), Sum(:x)]; key = :k,
                    keyset = ["a", "b"])),
        )
        @test df.time == [5, 5, 10, 10, 15, 15]
        @test df.count == [1, 0, 0, 2, 0, 0]
        @test df.x_sum == [1.0, 0.0, 0.0, 4.5, 0.0, 0.0]

        # the key column takes the declared type, matching isequal data keys
        r = onechunk(time = [1], k = [2], x = [1])
        df = DataFrame(
            load(Context(0, 10),
                r |> intervalize(clock(5), Count(); key = :k, keyset = [1.0, 2.0])),
        )
        @test df.k == [1.0, 2.0]
        @test eltype(df.k) == Float64
        @test df.count == [0, 1]

        # no data still emits the whole grid, typed from the configs
        empty = CausalPipeline(ctx -> DataFrame[])
        df = DataFrame(
            load(Context(0, 15),
                empty |> intervalize(clock(5), [Count(), Mean(:x)]; key = :k,
                    keyset = ["b", "a"], closelast = true)),
        )
        @test df.time == [5, 5, 10, 10, 15, 15]
        @test df.k == ["b", "a", "b", "a", "b", "a"]
        @test df.count == zeros(Int, 6)
        @test all(ismissing, df.x_mean)

        # several key columns, declared as tuples or as named tuples (in any
        # field order)
        m = onechunk(time = [1, 2], k = ["a", "b"], j = [1, 2], x = [1, 2])
        for declared in ([("a", 1), ("b", 1), ("b", 2)],
            [(j = 1, k = "a"), (k = "b", j = 1), (k = "b", j = 2)])
            df = DataFrame(
                load(Context(0, 10),
                    m |> intervalize(clock(5), Count(); key = [:k, :j],
                        keyset = declared)),
            )
            @test names(df) == ["time", "k", "j", "count"]
            @test df.k == ["a", "b", "b"]
            @test df.j == [1, 1, 2]
            @test df.count == [1, 0, 1]
        end

        # with a sorted keyset, the dense rows holding data are exactly the
        # sparse output, over many intervals, empty ones and chunk boundaries
        times = cumsum(lcgsequence(21, 200, 3))
        labels = map(v -> ("a", "b", "c", "d")[v+1], lcgsequence(22, 200, 4))
        xs = lcgsequence(23, 200, 9)
        big = CausalPipeline(
            ctx -> [
                DataFrame(time = times[r], k = labels[r], x = xs[r])
                for r in (1:70, 71:140, 141:200)
            ],
        )
        ss = [Count(), Sum(:x), Max(:x)]
        ctx = Context(0, last(times) + 7)
        for closelast in (false, true)
            sparse = DataFrame(load(ctx,
                big |> intervalize(clock(10), ss; key = :k, closelast)))
            dense = DataFrame(
                load(ctx,
                    big |> intervalize(clock(10), ss; key = :k,
                        keyset = ["a", "b", "c", "d"], closelast)),
            )
            @test nrow(dense) == 4 * length(unique(dense.time))
            @test isequal(dense[dense.count .> 0, :], sparse)
        end

        # errors: keyset without key, duplicate or malformed declarations, and
        # a folded row whose key was not declared
        @test_throws ArgumentError intervalize(clock(5), Count(); keyset = ["a"])
        @test_throws ArgumentError intervalize(clock(5), Count(); key = :k,
            keyset = ["a", "a"])
        @test_throws ArgumentError intervalize(clock(5), Count(); key = [:k, :j],
            keyset = [("a",)])
        @test_throws ArgumentError intervalize(clock(5), Count(); key = [:k, :j],
            keyset = [(k = "a", i = 1)])
        @test_throws ArgumentError intervalize(clock(5), Count(); key = [:k, :j],
            keyset = ["a"])
        @test_throws ArgumentError load(Context(0, 15),
            p |> intervalize(clock(5), Count(); key = :k, keyset = ["a"]))
    end

    @testset "intervals and cycles span chunk boundaries" begin
        # the interval [5, 10) is split across two chunks (t = 7 and t = 8)
        p = multichunk([DataFrame(time = [2, 7], x = [10, 20]),
            DataFrame(time = [8, 12], x = [30, 40])])
        df = DataFrame(
            load(Context(0, 15),
                p |> intervalize(clock(5), [Count(), Sum(:x)]; closelast = true)),
        )
        @test df.time == [5, 10, 15]
        @test df.count == [1, 2, 1]                # [5, 10) sums t = 7 and t = 8
        @test df.x_sum == [10, 50, 40]
    end

    @testset "schema widening carries the open interval and widens output" begin
        # Int in the first chunk, Float64 in the second; the [0, 5) interval
        # spans the promotion
        p = multichunk([DataFrame(time = [1], x = [10]),
            DataFrame(time = [6], x = [2.5])])
        df = DataFrame(
            load(Context(0, 10),
                p |> intervalize(clock(5), Sum(:x); closelast = true)),
        )
        @test df.time == [5, 10]
        @test df.x_sum == [10.0, 2.5]
        @test eltype(df.x_sum) == Float64
    end

    @testset "empty data stream still emits the keyless grid" begin
        empty = CausalPipeline(ctx -> DataFrame[])
        df = DataFrame(
            load(Context(0, 15),
                empty |> intervalize(clock(5), [Count(), Sum(:x)])),
        )
        @test df.time == [5, 10]
        @test df.count == [0, 0]
        @test df.x_sum == [0, 0]

        # keyed empty data emits nothing
        dfk = DataFrame(
            load(Context(0, 15),
                empty |> intervalize(clock(5), Count(); key = :k)),
        )
        @test nrow(dfk) == 0
    end

    @testset "edge cases: empty and single-boundary clocks" begin
        p = onechunk(time = [1, 2, 5, 8], x = [1, 2, 3, 4])
        # an empty clock defines no interval -> no output
        emptyclk = CausalPipeline(ctx -> DataFrame[])
        @test nrow(DataFrame(load(Context(0, 10),
            p |> intervalize(emptyclk, Count())))) == 0
        # a single boundary defines only the trailing partial -> only closelast
        # can emit
        clk1 = onechunk(time = [0])
        @test nrow(DataFrame(load(Context(0, 10),
            p |> intervalize(clk1, Count())))) == 0
        df = DataFrame(
            load(Context(0, 10),
                p |> intervalize(clk1, [Count(), Sum(:x)]; closelast = true)),
        )
        @test df.time == [10]
        @test df.count == [4]                      # all rows in [0, 10)
        @test df.x_sum == [10]
    end

    @testset "irregular clock; extra clock columns ignored" begin
        clk = onechunk(time = [0, 3, 9], label = ["a", "b", "c"])
        p = onechunk(time = [1, 2, 5, 8], x = [1, 2, 3, 4])
        df = DataFrame(load(Context(0, 10),
            p |> intervalize(clk, [Count(), Sum(:x)])))
        # [0, 3) -> 3: t = 1, 2; [3, 9) -> 9: t = 5, 8; trailing [9, 10) dropped
        @test df.time == [3, 9]
        @test df.count == [2, 2]
        @test df.x_sum == [3, 7]
    end

    @testset "curried and uncurried forms agree" begin
        p = onechunk(time = [2, 7], x = [10, 20])
        a = DataFrame(load(Context(0, 10), p |> intervalize(clock(5), Sum(:x))))
        b = DataFrame(load(Context(0, 10), intervalize(p, clock(5), Sum(:x))))
        @test a == b
    end

    @testset "streaming equals loading" begin
        p = onechunk(time = [2, 7, 7, 12], x = [10, 20, 30, 40])
        pipe = p |> intervalize(clock(5), [Count(), Sum(:x)]; closelast = true)
        loaded = DataFrame(load(Context(0, 15), pipe))
        streamed = reduce(vcat, DataFrame.(collect(stream(Context(0, 15), pipe))))
        @test loaded == streamed
    end

    @testset "validation" begin
        p = onechunk(time = [1], x = [1])
        # duplicate / :time keys rejected eagerly
        @test_throws ArgumentError intervalize(clock(5), Count(); key = [:k, :k])
        @test_throws ArgumentError intervalize(clock(5), Count(); key = :time)
        # a summarizer output column named :time is rejected (via prototypes)
        @test_throws ArgumentError intervalize(clock(5), BadTime())
        # a missing key column errors on the first chunk
        @test_throws ArgumentError DataFrame(
            load(Context(0, 10),
                p |> intervalize(clock(5), Count(); key = :nope)),
        )
    end
end
