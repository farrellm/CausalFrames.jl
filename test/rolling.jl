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

        # self-summarization runs the pipeline once, over the widened
        # context; without `sharedrun`, a second time as is for the rows
        ctxs = []
        selfrec = CausalPipeline() do ctx
            push!(ctxs, ctx)
            [DataFrame(time = [max(ctx.start, 3)], x = [1])]
        end
        load(Context(3, 10), selfrec |> addrollingcolumns((w2 = 2,), Count()))
        @test ctxs == [Context(1, 10)]
        empty!(ctxs)
        load(Context(3, 10),
            selfrec |> addrollingcolumns((w2 = 2,), Count(); sharedrun = false))
        @test sort(ctxs; by = c -> c.start) == [Context(1, 10), Context(3, 10)]
    end

    @testset "stored rows keep only the columns read" begin
        df = DataFrame(time = 1:6, k = repeat(["a", "b"], 3),
            x = Float64.(1:6), u = fill(missing, 6), v = string.(1:6))
        t(ss; kw...) = addrollingcolumns((w2 = 2, b2 = Bars(2)), ss; kw...)
        declared = Recording(Sum(:x), (:x,))
        load(Context(0, 10), readtable(df) |> t([declared]; key = :k))
        @test !isempty(declared.seen)
        @test all(==((:time, :k, :x)), declared.seen)
        # a summarizer that declares nothing keeps every column, for all
        undeclared = Recording(Sum(:x), nothing)
        load(Context(0, 10), readtable(df) |> t([undeclared, Count()]))
        @test !isempty(undeclared.seen)
        @test all(==((:time, :k, :x, :u, :v)), undeclared.seen)
        # a row term is kept under its name, after the columns it reads
        termed = Recording(Count(), (:r,))
        load(Context(0, 10),
            readtable(df) |> t([termed, Sum(:r => row -> 2 * row.x)]))
        @test !isempty(termed.seen)
        @test all(==((:time, :r)), termed.seen)
        # the output is still every input column plus the summaries
        out = DataFrame(load(Context(0, 10), readtable(df) |> t(Sum(:x))))
        @test names(out) == ["time", "k", "x", "u", "v", "w2_x_sum", "b2_x_sum"]
        @test out.w2_x_sum == [1.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    end

    @testset "self-summarization shares one run" begin
        df = DataFrame(time = repeat(1:30, inner = 2),
            k = repeat(["a", "b"], 30), x = Float64.(mod.(1:60, 7)))
        runs = Ref(0)
        counted = CausalPipeline(ctx -> (runs[] += 1; readtable(df).run(ctx)))
        chain(lb, depth; kw...) = foldl(1:depth; init = counted) do p, n
            p |> addrollingcolumns(NamedTuple{(Symbol(:w, n),)}((lb(n),)),
                Sum(:x); key = :k, kw...)
        end
        ctx = Context(5, 100)
        for lb in (Bars, identity), depth in 1:4
            runs[] = 0
            shared = DataFrame(load(ctx, chain(lb, depth)))
            @test runs[] == 1
            runs[] = 0
            unshared = DataFrame(load(ctx, chain(lb, depth; sharedrun = false)))
            # Without a time look-back the contexts agree, so the run is
            # shared either way; otherwise each stage runs its input twice.
            @test runs[] == (lb === Bars ? 1 : 2^depth)
            # Sum(:x) over the source's own column doesn't depend on history
            # before the window, so the two settings agree.
            @test isequal(shared, unshared)
            @test minimum(shared.time) == 5
        end

        # An input whose rows depend on history (a Bars stage) differs: shared,
        # the output shows the widened run the summaries were computed over.
        p = readtable(df) |> addrollingcolumns((b2 = Bars(2),), Sum(:x); key = :k)
        t(sr) = addrollingcolumns((w1 = 1,), Count(); key = :k, sharedrun = sr)
        shared = DataFrame(load(ctx, p |> t(true)))
        unshared = DataFrame(load(ctx, p |> t(false)))
        @test all(ismissing, unshared.b2_x_sum[1:2])
        @test !any(ismissing, shared.b2_x_sum)
        @test shared.b2_x_sum ==
              DataFrame(load(Context(4, 100), p)).b2_x_sum[3:end]
        @test shared.w1_count == unshared.w1_count

        # one chunk per row: a timestamp straddling chunks still has its
        # later rows in the earlier row's window, and lead-in chunks are dropped
        rowchunks = CausalPipeline(
            ctx -> (
                c for c in
                (df[i:i, :] for i in 1:nrow(df))
                if ctx.start <= c.time[1] < ctx.stop
            ),
        )
        whole = CausalPipeline(ctx -> [df[ctx.start .<= df.time .< ctx.stop, :]])
        rt = addrollingcolumns((w3 = 3, b2 = Bars(2)), Sum(:x))
        @test isequal(DataFrame(load(ctx, rowchunks |> rt)),
            DataFrame(load(ctx, whole |> rt)))
        @test isequal(reduce(vcat, DataFrame.(stream(ctx, rowchunks |> rt))),
            DataFrame(load(ctx, whole |> rt)))

        # The summarized side reads the lead-in before the output side pulls,
        # so each lead-in chunk is freed once its rows are admitted: the tee
        # never queues the lead-in, a look-back of chunks at full width.
        # Before each chunk, a full GC counts the lead-in chunks still alive.
        seq = DataFrame(time = 1:400, x = Float64.(1:400))
        leadin = WeakRef[]
        alive = Int[]
        tracked = CausalPipeline() do ctx
            rows = findall(ctx.start .<= seq.time .< ctx.stop)
            return (
                begin
                    GC.gc()
                    push!(alive, count(w -> w.value !== nothing, leadin))
                    c = seq[r, :]
                    last(c.time) < 300 && push!(leadin, WeakRef(c.x))
                    c
                end for r in Iterators.partition(rows, 10)
            )
        end
        out = DataFrame(
            load(Context(300, 401),
                tracked |> addrollingcolumns((w = 200,), Sum(:x))),
        )
        @test out.w_x_sum == [sum(Float64, (t-200):t) for t in 300:400]
        @test length(leadin) == 20
        @test maximum(alive) <= 3
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
        # a shared run over a widened context checks its augmented side on
        # the first chunk, before reading the lead-in
        @test_throws ArgumentError(
            "addrollingcolumns output column :w1_x_sum collides with an existing column",
        ) load(Context(0, 10), taken |> addrollingcolumns((w1 = 5,), Sum(:x)))
        @test_throws ArgumentError(
            "addrollingcolumns key column :k not found in the augmented input",
        ) load(Context(0, 10), p |> addrollingcolumns((w1 = 5,), Sum(:x); key = :k))
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

    # Column-wise agreement, exact but for floats, which may carry sliding-sum
    # round-off on the running path (and NaN, e.g. a single-row corrected
    # variance, which isapprox rejects). One assertion per comparison, naming
    # every column that disagreed.
    function agrees(fast, slow)
        @test names(fast) == names(slow)
        same(x, y) =
            ismissing(x) ? ismissing(y) :
            !ismissing(y) && (isapprox(x, y) || (isnan(x) && isnan(y)))
        colagrees(a, b) =
            nonmissingtype(eltype(a)) <: AbstractFloat ? all(map(same, a, b)) :
            isequal(a, b)
        @test isempty([n for n in names(fast) if !colagrees(fast[!, n], slow[!, n])])
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
    # `z` is never summarized, so the built-ins' stored rows project it out
    # while the Opaque oracle, declaring no input columns, keeps it.
    zs = [isodd(i) ? missing : Float64(i) for i in 1:nrows]
    mkdata(x) = CausalPipeline(
        ctx ->
            [
                DataFrame(time = times[r], k = ks[r], x = x[r], y = ys[r],
                    z = zs[r]) for r in ranges
            ],
    )
    intdata = mkdata(xs)
    floatdata = mkdata(Float64.(xs) ./ 4)
    # same streams with missing sprinkled through the summarized column, so the
    # sum accumulators go through their counting states on the running path
    xsm = [i % 5 == 0 ? missing : xs[i] for i in 1:nrows]
    missingintdata = mkdata(convert(Vector{Union{Missing,Int}}, xsm))
    missingfloatdata = mkdata(
        convert(Vector{Union{Missing,Float64}}, map(v -> ismissing(v) ? v : v / 4, xsm)))

    windows = (w0 = 0, w3 = 3, w20 = 20)
    rolled(p, ss; kwargs...) = DataFrame(load(Context(0, 1000),
        p |> addrollingcolumns(windows, ss; kwargs...)))

    # The sets exercise each window tier (tiers.jl) alone and together.
    # DotProduct(:y, :x) and Covariance(:y, :x) use the reversed argument
    # order, so they fold through the alias over the canonical accumulator, on
    # the running tier. The trackers and CountDistinct use their windowed states.
    groupset = [Count(), Sum(:x), Mean(:x), Variance(:x), Correlation(:x, :y),
        DotProduct(:y, :x), Covariance(:y, :x), AgeWeightedSum(:x),
        LinearRegression(:x, :y; name = :m1),
        LinearRegression([:x, :y], :y; name = :m2)]
    trackset = [Min(:x), Max(:x), First(:x), Last(:x), CountDistinct(:x)]
    # the arg-extremes slide the same deque; the small integer pools tie often
    indexset = [MinIndex(:x), MaxIndex(:x), MinWithIndex(:y), MaxWithIndex(:y)]
    monoidset = [Product(:x), MinMax(:y)]                 # the tree alone
    mixedset = [Sum(:x), Min(:x), MinMax(:y), Product(:x)] # running + tree
    plainset = [Sum(:x), TestVar(:x), PlainSum(:y)]       # running + re-fold
    spanset = [TierSpan(:x), Mean(:y), Last(:y)]          # every tier at once
    # the sorted accumulator and its dependents, PercentRank through Last
    orderset = [Quantile(:x, [0.1, 0.5, 0.9]), Median(:x), PercentRank(:x), MeanAbsDev(:x),
        Quantile(:y, [0.07, 0.25, 1.0]; interpolation = :nearestrank)]
    allsets = (groupset, trackset, indexset, monoidset, mixedset, plainset,
        spanset, orderset)

    @testset "differential against the re-fold oracle" begin
        for p in (intdata, floatdata), ss in allsets

            agrees(rolled(p, ss), rolled(p, map(Opaque, ss)))
            agrees(rolled(p, ss; key = :k),
                rolled(p, map(Opaque, ss); key = :k))
        end
    end

    @testset "differential over missing inputs" begin
        # the running path must equal the re-fold oracle with missing in the
        # summarized column, which the counting states keep on the running path
        for p in (missingintdata, missingfloatdata),
            ss in (groupset, trackset, indexset, mixedset, spanset, orderset)

            agrees(rolled(p, ss), rolled(p, map(Opaque, ss)))
            agrees(rolled(p, ss; key = :k),
                rolled(p, map(Opaque, ss); key = :k))
        end
    end

    @testset "the tree tier agrees with the running one" begin
        # AsMonoid hides the groups down to monoids, so the sorted accumulator
        # and the arg-extremes merge through the segment tree's combine!
        # instead of sliding
        for p in (intdata, floatdata, missingfloatdata), key in (nothing, :k),
            ss in (orderset, indexset)

            agrees(rolled(p, ss; key), rolled(p, map(AsMonoid, ss); key))
        end
    end

    @testset "streaming agrees with loading" begin
        for ss in (groupset, trackset, indexset, spanset, orderset)
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

    @testset "missing rows recover in running mode" begin
        # chunk two widens the sum accumulator to admit missing, which it counts
        # rather than folds (like NaN/±Inf), so it stays on the running path:
        # the missing row's windows report missing, and later windows recover
        # exactly
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
        # a float accumulator counts NaN and ±Inf inputs rather than folding
        # them, so it stays on the running path and the window recovers exactly
        # once the nonfinite row expires (a naive running sum stays NaN)
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

    @testset "order statistics recover in running mode" begin
        # missing and NaN are counted, not stored, so either poisons only the
        # windows holding its row
        for bad in (NaN, missing)
            p = onechunk(time = [1, 2, 3, 10],
                y = Union{Missing,Float64}[1.0, 2.0, bad, 5.0])
            df = DataFrame(
                load(Context(0, 20),
                    p |> addrollingcolumns((w2 = 2,), [Median(:y), PercentRank(:y)])),
            )
            @test isequal(df.w2_y_median, [1.0, 1.5, bad, 5.0])
            @test isequal(df.w2_y_percentrank, [NaN, 1.0, bad, NaN])
        end
    end

    @testset "compensated sliding accuracy" begin
        # once the large value leaves the window only the compensation term
        # remains; a naive running sum would give 0.0, which isapprox rejects
        p = onechunk(time = [1, 2, 3], x = [1e16, 1.0, 1.0])
        df = DataFrame(load(Context(0, 10),
            p |> addrollingcolumns((w1 = 1,), Sum(:x))))
        @test df.w1_x_sum[3] == 2.0
    end

    @testset "order sensitivity through ties" begin
        # First/Last through ties: every row at time t sees all rows tied at
        # t, in stream order, on the running tier and the re-fold oracle alike
        p = onechunk(time = [1, 2, 2, 3], x = [1, 2, 3, 4])
        for ss in ([First(:x), Last(:x)], [Opaque(First(:x)), Opaque(Last(:x))])
            df = DataFrame(
                load(Context(0, 10), p |> addrollingcolumns((w1 = 1,), ss)))
            @test isequal(df.w1_x_first, [1, 1, 1, 2])
            @test isequal(df.w1_x_last, [1, 3, 3, 4])
        end
    end

    @testset "fast paths widen like the re-fold path" begin
        # Int then Float64 across summarized chunks, on both fast paths: the
        # states, buffer and half-filled value vectors widen mid-augmented-chunk
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
                p |> addrollingcolumns((w5 = 5,), [Sum(:y), Min(:y), Product(:y)];
                    from = src)),
        )
        @test df.w5_y_sum == [1.0, 3.0, 5.5]
        @test isequal(df.w5_y_min, [1.0, 1.0, 1.0])
        @test df.w5_y_product == [1.0, 2.0, 5.0]
    end

    @testset "tiers are chosen per accumulator" begin
        tiers(ss, types) = CausalFrames.tiernames(
            CausalFrames.tiering(
                first(CausalFrames.prototypes(CausalFrames.tosummarizers(ss),
                        Symbol[])), types),
        )
        it = (time = Int, k = String, x = Int, y = Int)
        # in expansion order: TierSpan's dependencies, TierSpan (fieldless, so
        # no tier), then Mean's Count and Sum, Mean, and Last
        @test tiers(spanset, it) == (:running, :tree, :refold, :derived,
            :running, :running, :derived, :running)
        # the sorted accumulator is a group, and so is Last beside it
        @test tiers([PercentRank(:x), Quantile(:x, 0.5)], it) ==
              (:running, :running, :derived, :derived)
        @test tiers(map(AsMonoid, [PercentRank(:x)]), it) ==
              (:tree, :tree, :derived)
        # the arg-extremes slide their deque
        @test tiers(indexset, it) == (:running, :running, :running, :running)
        # Last is a group, so beside Mean the whole call slides
        @test tiers([Last(:x), Mean(:x)], it) ==
              (:running, :running, :running, :derived)
        # a widening that defeats isinvertible demotes only its accumulator
        @test tiers([FragileSum(:x), Sum(:y)], it) == (:running, :running)
        @test tiers([FragileSum(:x), Sum(:y)], merge(it, (; x = Float64))) ==
              (:tree, :running)

        # Last beside Mean, keyed, over ties and windows that empty and refill
        for p in (intdata, floatdata)
            agrees(rolled(p, [Last(:x), Mean(:x)]; key = :k),
                rolled(p, map(Opaque, [Last(:x), Mean(:x)]); key = :k))
        end

        # the demotion mid-stream, beside each other tier: x turns Float64 in
        # the second chunk, so FragileSum rebuilds into the tree from the
        # buffer (beside a running Sum), from the previous trees (beside
        # Product), or beside a refold PlainSum
        widening = CausalPipeline(
            ctx -> [
                DataFrame(time = times[r], k = ks[r],
                    x = r == 1:100 ? xs[r] : Float64.(xs[r]) ./ 4, y = ys[r])
                for r in ranges
            ])
        # LateSum goes the other way, promoted from a tree-only call that kept
        # no buffer, so the running tier gathers one from the previous trees
        @test tiers([LateSum(:x)], it) == (:tree,)
        @test tiers([LateSum(:x)], merge(it, (; x = Float64))) == (:running,)
        for ss in ([FragileSum(:x), Sum(:y), Last(:x)],
                [FragileSum(:x), Product(:y)], [FragileSum(:x), PlainSum(:y)],
                [LateSum(:x)], [LateSum(:x), Product(:y)]),
            key in (nothing, :k)

            agrees(rolled(widening, ss; key), rolled(widening, map(Opaque, ss); key))
        end
    end

    @testset "allocations do not grow with rows" begin
        # Per tier and mixed: running groups are pooled, windowed states and
        # trees reuse their storage, and the refold tier threads one scratch
        # tuple, so quadrupling the rows adds only logarithmic buffer growth.
        # (Not MinMax: its `fresh!` is the allocating default, which a tree
        # query calls per window.)
        function rollallocs(ss, n, windows = (w5 = 5, w50 = 50))
            src = DataFrame(time = 1:n, k = repeat(["a", "b"], n ÷ 2),
                x = mod.(1:n, 7), y = mod.(1:n, 5))
            p = CausalPipeline(ctx -> [src])
            t = addrollingcolumns(windows, ss; key = :k)
            load(Context(0, n + 1), p |> t)
            return @allocations load(Context(0, n + 1), p |> t)
        end
        for ss in ([Sum(:x), Mean(:x), Last(:x), Min(:x), CountDistinct(:x),
            AgeWeightedSum(:x)], [Product(:x)], [PlainSum(:x)],
            spanset, orderset)

            @test rollallocs(ss, 8000) - rollallocs(ss, 2000) < 200
            barwins = (b5 = Bars(5), w5 = 5, b50 = Bars(50))
            @test rollallocs(ss, 8000, barwins) - rollallocs(ss, 2000, barwins) <
                  200
        end

        # A summarized input wider than 32 columns, where Base's tuple `map`
        # stops unrolling, still admits rows without allocating (issue #98).
        function wideallocs(n)
            src = DataFrame(time = 1:n, k = repeat(["a", "b"], n ÷ 2),
                x = mod.(1:n, 7))
            for j in 1:40
                src[!, "c$j"] = fill(1.0, n)
            end
            p = CausalPipeline(ctx -> [src])
            t = addrollingcolumns((w5 = 5, b5 = Bars(5)), Sum(:x); key = :k)
            load(Context(0, n + 1), p |> t)
            return @allocations load(Context(0, n + 1), p |> t)
        end
        @test wideallocs(8000) - wideallocs(2000) < 200
    end

    @testset "fast paths over dates and mixed periods" begin
        t0 = DateTime(2024, 1, 1)
        p = onechunk(time = [t0, t0 + Minute(30), t0 + Minute(62)],
            x = [1, 2, 3])
        # (the running path's Sum over the same data is pinned in "dates and
        # mixed periods")
        df = DataFrame(
            load(Context(t0, t0 + Hour(2)),
                p |> addrollingcolumns((m5 = Minute(5), h1 = Hour(1)),
                    [Min(:x), Sum(:x)])),
        )
        @test df.h1_x_min == [1, 1, 2]
    end
    @testset "Bars windows" begin
        bars = (b1 = Bars(1), b4 = Bars(4), b25 = Bars(25))
        barred(p, ss; kwargs...) = DataFrame(load(Context(0, 1000),
            p |> addrollingcolumns(bars, ss; kwargs...)))

        # The independent oracle: each row's window is the last n rows at or
        # before its time (under its key), all missing until there are n.
        function barsoracle(p, ss; key = nothing)
            df = DataFrame(load(Context(0, 1000), p))
            protos, requested =
                CausalFrames.prototypes(CausalFrames.tosummarizers(ss), Symbol[])
            types = map(eltype, Tables.columntable(df))
            rows = Tables.rowtable(df)
            out = copy(df)
            for (w, lb) in pairs(bars)
                vals = map(eachindex(rows)) do i
                    idx = [
                        j for j in eachindex(rows)
                        if rows[j].time <= rows[i].time &&
                            (key === nothing || isequal(rows[j][key], rows[i][key]))
                    ]
                    length(idx) < lb.n && return nothing
                    states = CausalFrames.newstates(protos, types)
                    foreach(j -> CausalFrames.updateall!(states, rows[j]),
                        idx[(end-lb.n+1):end])
                    return CausalFrames.summaryvalues(states, Val(requested))
                end
                for n in requested
                    out[!, Symbol(w, '_', n)] =
                        [v === nothing ? missing : v[n] for v in vals]
                end
            end
            return out
        end

        @testset "differential against the re-fold oracle" begin
            for p in (intdata, floatdata, missingfloatdata), ss in allsets
                agrees(barred(p, ss), barred(p, map(Opaque, ss)))
                agrees(barred(p, ss; key = :k),
                    barred(p, map(Opaque, ss); key = :k))
            end
        end

        @testset "against the brute-force oracle" begin
            for p in (intdata, missingfloatdata),
                ss in ([Sum(:x), Mean(:x), First(:x), Last(:x)],
                    [Product(:x), Max(:x)])

                for key in (nothing, :k)
                    agrees(barred(p, ss; key), barsoracle(p, ss; key))
                    agrees(barred(p, map(Opaque, ss); key), barsoracle(p, ss; key))
                end
            end
        end

        @testset "partial windows are missing, typed Union{Missing, T}" begin
            p = onechunk(time = [1, 2, 3], x = [1, 2, 3])
            df = DataFrame(
                load(Context(0, 10),
                    p |> addrollingcolumns(:b2 => Bars(2), [Count(), Sum(:x)])),
            )
            @test isequal(df.b2_count, [missing, 2, 2])
            @test isequal(df.b2_x_sum, [missing, 3, 5])
            @test eltype(df.b2_count) == Union{Missing,Int}
            @test eltype(df.b2_x_sum) == Union{Missing,Int}
            # beside a time window, whose empty values are not missing
            df = DataFrame(
                load(Context(0, 10),
                    p |> addrollingcolumns((b2 = Bars(2), t0 = 0), Count())),
            )
            @test isequal(df.b2_count, [missing, 2, 2])
            @test df.t0_count == [1, 1, 1]
            @test eltype(df.t0_count) == Int
        end

        @testset "uniformly spaced rows match a time look-back" begin
            p = onechunk(time = 1:40, x = map(v -> v - 3, lcgsequence(5, 40, 7)))
            ss = [Sum(:x), Mean(:x), Min(:x), Product(:x), PlainSum(:x)]
            df = DataFrame(
                load(Context(0, 100),
                    p |> addrollingcolumns((b5 = Bars(5), t4 = 4), ss)),
            )
            for n in ("x_sum", "x_mean", "x_min", "x_product", "x_plainsum")
                @test all(ismissing, df[1:4, "b5_"*n])
                @test isequal(df[5:end, "b5_"*n], df[5:end, "t4_"*n])
            end
        end

        @testset "mixed with time windows" begin
            for p in (intdata, floatdata), ss in (mixedset, spanset),
                key in (nothing, :k)

                mixed = DataFrame(
                    load(Context(0, 1000),
                        p |> addrollingcolumns((w3 = 3, b4 = Bars(4), w20 = 20), ss;
                            key)),
                )
                timed = rolled(p, ss; key)
                barsonly = barred(p, ss; key)
                agrees(select(mixed, r"^(time|k|x|y|w3_|w20_)"),
                    select(timed, r"^(time|k|x|y|w3_|w20_)"))
                agrees(select(mixed, r"^b4_"), select(barsonly, r"^b4_"))
            end
        end

        @testset "ties" begin
            # a tie's later rows are in the earlier row's window
            p = onechunk(time = [1, 1, 2, 3, 3, 4, 5],
                k = [:a, :b, :a, :a, :b, :b, :a], x = 1.0:7.0)
            for ss in ([Sum(:x), First(:x)], [Opaque(Sum(:x)), Opaque(First(:x))],
                [AsMonoid(Sum(:x)), AsMonoid(First(:x))])

                df = DataFrame(
                    load(Context(0, 10),
                        p |> addrollingcolumns(:b2 => Bars(2), ss)),
                )
                @test isequal(df.b2_x_sum, [3.0, 3.0, 5.0, 9.0, 9.0, 11.0, 13.0])
                @test isequal(df.b2_x_first, [1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 6.0])
                df = DataFrame(
                    load(Context(0, 10),
                        p |> addrollingcolumns(:b2 => Bars(2), ss; key = :k)),
                )
                @test isequal(df.b2_x_sum,
                    [missing, missing, 4.0, 7.0, 7.0, 11.0, 11.0])
                @test isequal(df.b2_x_first,
                    [missing, missing, 1.0, 3.0, 2.0, 5.0, 4.0])
            end
        end

        @testset "from counts the summarized stream's rows" begin
            p = onechunk(time = [2, 4, 6], k = [:a, :a, :b])
            src = onechunk(time = [1, 1, 3, 3, 5], k = [:a, :b, :a, :a, :b],
                x = [1, 10, 2, 3, 20])
            for ss in ([Sum(:x)], [Opaque(Sum(:x))], [AsMonoid(Sum(:x))])
                df = DataFrame(
                    load(Context(0, 10),
                        p |> addrollingcolumns(:b2 => Bars(2), ss; key = :k,
                            from = src)),
                )
                @test isequal(df.b2_x_sum, [missing, 5, 30])
            end
        end

        @testset "nonfinite and missing rows recover on expiry" begin
            p = onechunk(time = 1:6, x = [1.0, NaN, 2.0, 3.0, Inf, 4.0])
            q = onechunk(time = 1:5, x = [1.0, missing, 2.0, 3.0, 4.0])
            for ss in ([Sum(:x)], [Opaque(Sum(:x))], [AsMonoid(Sum(:x))])
                df = DataFrame(
                    load(Context(0, 10),
                        p |> addrollingcolumns(:b2 => Bars(2), ss)),
                )
                @test isequal(df.b2_x_sum, [missing, NaN, NaN, 5.0, Inf, Inf])
                df = DataFrame(
                    load(Context(0, 10),
                        q |> addrollingcolumns(:b2 => Bars(2), ss)),
                )
                @test isequal(df.b2_x_sum, [missing, missing, missing, 5.0, 7.0])
            end
        end

        @testset "the context is not widened" begin
            p = readtable(DataFrame(time = 1:6, x = 1:6))  # clipped to the context
            df = DataFrame(
                load(Context(4, 10),
                    p |> addrollingcolumns(:b2 => Bars(2), Sum(:x))),
            )
            @test isequal(df.b2_x_sum, [missing, 9, 11])
            # a time window in the same call widens the one summarized stream,
            # and a Bars window counts whatever rows it sees
            df = DataFrame(
                load(Context(4, 10),
                    p |> addrollingcolumns((b2 = Bars(2), t1 = 1), Sum(:x))),
            )
            @test df.b2_x_sum == [7, 9, 11]
            @test df.t1_x_sum == [7, 9, 11]
        end

        @testset "widening mid-stream" begin
            widening = CausalPipeline(
                ctx -> [
                    DataFrame(time = times[r], k = ks[r],
                        x = r == 1:100 ? xs[r] : Float64.(xs[r]) ./ 4, y = ys[r])
                    for r in ranges
                ])
            for ss in ([Sum(:x), Mean(:x)], [FragileSum(:x), Sum(:y), Last(:x)],
                    [FragileSum(:x), Product(:y)], [FragileSum(:x), PlainSum(:y)],
                    [LateSum(:x)], [LateSum(:x), Product(:y)]),
                windows in (bars, (w3 = 3, b4 = Bars(4))),
                key in (nothing, :k)

                agrees(
                    DataFrame(
                        load(Context(0, 1000),
                            widening |> addrollingcolumns(windows, ss; key)),
                    ),
                    DataFrame(
                        load(Context(0, 1000),
                            widening |> addrollingcolumns(windows, map(Opaque, ss);
                                key)),
                    ))
            end
        end

        @testset "an empty summarized stream" begin
            p = onechunk(time = [1, 2], x = [1, 2])
            df = DataFrame(
                load(Context(0, 10),
                    p |> addrollingcolumns((b2 = Bars(2), w2 = 2), Sum(:x);
                        from = emptyframe())),
            )
            @test all(ismissing, df.b2_x_sum)
            @test eltype(df.b2_x_sum) == Union{Missing,Int}
            @test df.w2_x_sum == [0, 0]
        end

        @testset "validation" begin
            @test_throws ArgumentError("Bars count must be positive, got 0") Bars(0)
            @test_throws ArgumentError Bars(-3)
            @test Bars(Int8(3)).n === 3
            @test_throws ArgumentError(
                "summarizewindows lookback must be a time span, got Bars(2); " *
                "`Bars` look-backs are for `addrollingcolumns`",
            ) summarizewindows(clock(1), Bars(2), Sum(:x))
        end
    end
end
