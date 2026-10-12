@testset "row terms" begin
    chunks = [
        DataFrame(time = [1, 2, 2], k = ["a", "b", "a"], high = [5.0, 6.0, 8.0],
            low = [4.0, 4.5, 5.0], v = [1, 2, 3]),
        DataFrame(time = [4, 5, 7], k = ["b", "a", "b"], high = [9.0, 9.5, 7.0],
            low = [7.0, 1.0, 6.5], v = [4, 5, 6]),
    ]
    src = CausalPipeline(_ -> map(copy, chunks))
    rng = rangeterm
    added = src |> addcolumns(r -> (; range = rng(r)))
    ctx = Context(0, 10)
    run(p) = DataFrame(load(ctx, p))

    @testset "constructors" begin
        @test Sum(:x) === Sum{:x}()
        @test DotProduct(:x, :y) === DotProduct{:x,:y}()
        s = Sum(:range => rng)
        @test s isa CausalFrames.Termed
        @test CausalFrames.unterm(s) === Sum{:range}()
        @test CausalFrames.rowterms(s) === (; range = rng)
        @test keys(CausalFrames.emptyvalue(s)) == (:range_sum,)
        d = DotProduct(:v, :range => rng)
        @test CausalFrames.unterm(d) === DotProduct{:v,:range}()
        @test CausalFrames.rowterms(DotProduct(:range => rng, :range => rng)) ===
              (; range = rng)
    end

    @testset "matches addcolumns then the summarizer" begin
        ss(term) = [Sum(term), DotProduct(:v, term), DotProduct(term, :v)]
        for key in (nothing, :k)
            @test isequal(run(src |> summarize(ss(:range => rng); key)),
                run(added |> summarize(ss(:range); key)))
            @test isequal(run(src |> summarizecycles(ss(:range => rng); key)),
                run(added |> summarizecycles(ss(:range); key)))
            @test isequal(
                run(src |> addsummarycolumns(ss(:range => rng); key)),
                select(run(added |> addsummarycolumns(ss(:range); key)), Not(:range)),
            )
            @test isequal(
                run(src |> addrollingcolumns((w2 = 2,), ss(:range => rng); key)),
                select(run(added |> addrollingcolumns((w2 = 2,), ss(:range); key)),
                    Not(:range)),
            )
            @test isequal(
                run(src |> intervalize(clock(3), ss(:range => rng); key)),
                run(added |> intervalize(clock(3), ss(:range); key)),
            )
            @test isequal(
                run(src |> summarizewindows(clock(3), 4, ss(:range => rng); key)),
                run(added |> summarizewindows(clock(3), 4, ss(:range); key)),
            )
        end
        @test isequal(
            run(src |> summarizecycles(ss(:range => rng); key = :k, keyset = ["a", "b"])),
            run(added |> summarizecycles(ss(:range); key = :k, keyset = ["a", "b"])),
        )
    end

    @testset "output names and shared state" begin
        df = run(
            src |> summarize([DotProduct(:v, :range => rng),
                DotProduct(:range => rng, :v)]),
        )
        @test names(df) == ["time", "v_range_dotproduct", "range_v_dotproduct"]
        @test df.v_range_dotproduct == df.range_v_dotproduct
        # One accumulator: the reversed form aliases the canonical one.
        protos, _, terms = CausalFrames.prototypes(
            Summarizer[DotProduct(:v, :range => rng), DotProduct(:range => rng, :v)],
            Symbol[])
        @test protos == (DotProduct{:range,:v}(), DotProduct{:v,:range}())
        @test terms === (; range = rng)
        # A term reused by a plain name (here a dependency) is the same column.
        df = run(src |> summarize([Sum(:range => rng), Sum(:range)]))
        @test names(df) == ["time", "range_sum"]
    end

    @testset "dependencies declare row terms" begin
        df = run(src |> summarize(RangeTotal()))
        @test names(df) == ["time", "rangetotal"]
        @test df.rangetotal == run(added |> summarize(Sum(:range))).range_sum
    end

    @testset "term columns never reach the output" begin
        @test !("range" in names(run(src |> addsummarycolumns(Sum(:range => rng)))))
        @test !(
            "range" in names(run(src |> addrollingcolumns(:w => 2,
                Sum(:range => rng))))
        )
    end

    @testset "term types widen like Sum" begin
        p = CausalPipeline(_ -> [DataFrame(time = 1:3, x = Int32[1, 2, 3])])
        df = run(p |> summarize(Sum(:y => r -> r.x * Int32(2))))
        @test df.y_sum == [12] && eltype(df.y_sum) == Int
        # A term whose values widen from Int to Float64 between chunks.
        q = CausalPipeline(
            _ -> [DataFrame(time = [1, 2], x = [1, 2]),
                DataFrame(time = [3], x = [0.5])],
        )
        df = run(q |> summarize(Sum(:y => r -> r.x)))
        @test df.y_sum == [3.5] && eltype(df.y_sum) == Float64
    end

    @testset "running path recovers from nonfinite and missing terms" begin
        for bad in (NaN, missing)
            p = CausalPipeline(
                _ -> [
                    DataFrame(time = [1, 2, 3, 10],
                        x = Union{Missing,Float64}[1.0, 2.0, bad, 5.0]),
                ],
            )
            twice = r -> 2 * r.x
            df = run(p |> addrollingcolumns((w2 = 2,),
                [Sum(:y => twice), Mean(:y)]))
            @test df.w2_y_sum[1:2] == [2.0, 6.0]
            @test isequal(df.w2_y_sum[3], bad === missing ? missing : NaN)
            @test df.w2_y_sum[4] == 10.0 && df.w2_y_mean[4] == 10.0
            protos, _, _ = CausalFrames.prototypes(
                Summarizer[Sum(:y => twice)], Symbol[])
            tg = CausalFrames.tiering(protos,
                (time = Int, x = Union{Missing,Float64}, y = Union{Missing,Float64}))
            @test length(tg.running) == 1
        end
    end

    @testset "errors" begin
        other = r -> r.high - r.low
        @test_throws "summarize row term :range is defined by more than one function" load(
            ctx, src |> summarize([Sum(:range => rng), Sum(:range => other)]))
        @test_throws "addrollingcolumns row term :range is defined by more than one function" load(
            ctx,
            src |> addrollingcolumns(:w => 2,
                [Sum(:range => rng), DotProduct(:v, :range => other)]),
        )
        @test_throws "DotProduct row term :range is defined by more than one function" DotProduct(
            :range => rng, :range => other)
        @test_throws "summarize row term :v collides with an input column" load(
            ctx, src |> summarize(Sum(:v => rng)))
        @test_throws "summarize row term :k collides with an input column" load(
            ctx, src |> summarize(Sum(:k => rng); key = :k))
        @test_throws "summarizewindows row term :low collides with an input column" load(
            ctx, src |> summarizewindows(clock(3), 4, Sum(:low => rng)))
        @test_throws "Sum row term may not be named :time" Sum(:time => rng)
        @test_throws MethodError Sum(:range => 1)
    end

    @testset "every column summarizer takes row terms" begin
        # Each form over a term must equal the same summarizer over the column
        # an `addcolumns` step made, across the running, tree and refold tiers.
        mid = r -> (r.high + r.low) / 2
        addedmid = src |> addcolumns(r -> (; range = rng(r), mid = mid(r)))
        forms = [
            (t -> CountDistinct(t), :range), (t -> SumPower(t, 3), :range),
            (t -> Product(t), :range), (t -> AgeWeightedSum(t), :range),
            (t -> Moment(t, 2), :range), (t -> Mean(t), :range),
            (t -> Variance(t), :range), (t -> Std(t; corrected = false), :range),
            (t -> CausalFrames.SortedValues(t), :range),
            (t -> CausalFrames.WindowValues(t), :range), (t -> MeanAbsDev(t), :range),
            (t -> Quantile(t, [0.25, 0.5]), :range),
            (t -> Quantile(t, 0.9; interpolation = :nearestrank), :range),
            (t -> Median(t), :range), (t -> PercentRank(t), :range),
            (t -> Min(t), :range), (t -> Max(t), :range),
            (t -> First(t), :range), (t -> Last(t), :range),
        ]
        for (make, c) in forms
            s = make(:range => rng)
            @test CausalFrames.unterm(s) == make(c)
            s isa CausalFrames.Termed{
                <:Union{CausalFrames.SortedValues,CausalFrames.WindowValues},
            } &&
                continue
            @test isequal(
                run(src |> addrollingcolumns((w2 = 2,), s; key = :k)),
                select(run(added |> addrollingcolumns((w2 = 2,), make(c); key = :k)),
                    Not(:range)),
            )
        end
        # Two-column forms, with the term in either position.
        for make in (DotProduct, Covariance, Correlation,
            (a, b) -> Covariance(a, b; corrected = false))
            for (a, b) in ((:range => rng, :mid => mid), (:mid => mid, :v),
                (:v, :range => rng))
                plain(x) = x isa Pair ? first(x) : x
                @test isequal(run(src |> summarize(make(a, b); key = :k)),
                    run(addedmid |> summarize(make(plain(a), plain(b)); key = :k)))
            end
        end
        # LinearRegression, with terms as predictors and as the response.
        lr = LinearRegression([:range => rng, :v], :mid => mid; name = :m)
        @test CausalFrames.rowterms(lr) === (; range = rng, mid = mid)
        @test isequal(run(src |> summarize(lr)),
            run(addedmid |> summarize(LinearRegression([:range, :v], :mid; name = :m))))
        lone = LinearRegression(:range => rng, :v)
        @test CausalFrames.unterm(lone) == LinearRegression(:range, :v)
        @test isequal(run(src |> summarize(lone)),
            run(added |> summarize(LinearRegression(:range, :v))))
        # A dependent over a term registers it once for its dependencies.
        protos, _, terms = CausalFrames.prototypes(
            Summarizer[Variance(:range => rng), Mean(:range => rng)], Symbol[])
        @test terms === (; range = rng)
        @test Sum{:range}() in protos
    end

    @testset "allocations do not grow with rows" begin
        function rollallocs(n)
            df = DataFrame(time = 1:n, k = repeat(["a", "b"], n ÷ 2),
                x = mod.(1:n, 7) .+ 0.5)
            p = CausalPipeline(_ -> [df])
            t = addrollingcolumns((w5 = 5,), [Sum(:y => r -> 2 * r.x)]; key = :k)
            load(Context(0, n + 1), p |> t)
            return @allocations load(Context(0, n + 1), p |> t)
        end
        @test rollallocs(8000) - rollallocs(2000) < 200
    end
end
