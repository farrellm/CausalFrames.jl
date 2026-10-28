@testset "shared runs" begin
    df = DataFrame(time = repeat(1:40, inner = 2), k = repeat(["a", "b"], 40),
        x = Float64.(mod.(1:80, 7)),
        m = [isodd(i ÷ 3) ? missing : Float64(i) for i in 1:80])
    # A context-free source counting its runs.
    function counting(df)
        runs = Ref(0)
        p = CausalPipeline(; startfree = true, stopfree = true) do ctx
            runs[] += 1
            return CausalFrames.runchunks(ctx, readtable(df))
        end
        return p, runs
    end
    ctx = Context(5, 35)
    # The input's `:x` under a new name, with the key, for a right side.
    renamed(n) =
        q ->
            q |> selectcolumns(:time, :k, :x) |>
            addcolumns(r -> NamedTuple{(Symbol(:x, n),)}((r.x,))) |> dropcolumns(:x)

    @testset "a diamond runs its upstream once" begin
        # Issue #105's table: d chained stages each reading their input on
        # both sides ran the source 2^d times.
        stages = (
            "asofjoin" => (p, n) -> asofjoin(p, p |> renamed(n); key = :k),
            "asofjoin tolerance" =>
                (p, n) -> asofjoin(p, p |> renamed(n); key = :k, tolerance = 2),
            "asofjoin lag" =>
                (p, n) -> asofjoin(p, p |> renamed(n) |> lag(1); key = :k),
            "merge" => (p, n) -> merge(p, p),
            "intervalize" => (p, n) -> intervalize(p, p, Count()),
            "summarizewindows" => (p, n) -> summarizewindows(p, p, 3, Count()),
            "addrollingcolumns from" =>
                (p, n) -> addrollingcolumns(p, NamedTuple{(Symbol(:w, n),)}((2,)),
                    Count(); from = p |> filterrows(r -> r.time > 0)),
            "addrollingcolumns" =>
                (p, n) -> addrollingcolumns(p, NamedTuple{(Symbol(:w, n),)}((2,)),
                    Count()),
            "warmup" => (p, n) -> warmup(p, 2, q -> q |> filterrows(r -> true)),
        )
        for (name, stage) in stages, depth in 1:5
            src, runs = counting(df)
            p = foldl(stage, 1:depth; init = src)
            load(ctx, p)
            @test runs[] == 1
        end

        # Unshared, the same stages compound.
        for depth in 1:4
            src, runs = counting(df)
            p = foldl(1:depth; init = src) do q, n
                q = q |> sharing(false)
                asofjoin(q, q |> renamed(n); key = :k, tolerance = 2)
            end
            load(ctx, p)
            @test runs[] == 2^depth
        end
    end

    @testset "a context-dependent input runs once per context" begin
        runs = Ref(0)
        dep = CausalPipeline() do c
            runs[] += 1
            return CausalFrames.runchunks(c, readtable(df))
        end
        @test !dep.startfree && !dep.stopfree
        chain(depth; force = nothing) = foldl(1:depth; init = dep) do q, n
            q = force === nothing ? q : q |> sharing(force)
            addrollingcolumns(q, NamedTuple{(Symbol(:w, n),)}((2,)), Count())
        end
        for depth in 1:4
            runs[] = 0
            load(ctx, chain(depth))
            # Each stage reads its input over two starts, but the runs asking
            # for equal contexts still share: one run per distinct start.
            @test runs[] == depth + 1
            runs[] = 0
            load(ctx, chain(depth; force = true))
            @test runs[] == 1
        end
        # Equal contexts share without any freedom: no look-back widens.
        runs[] = 0
        load(ctx, merge(dep, dep |> filterrows(r -> true)))
        @test runs[] == 1
        # and unshared, even those run apart
        runs[] = 0
        q = dep |> sharing(false)
        load(ctx, merge(q, q))
        @test runs[] == 2
    end

    @testset "flags" begin
        # the rest are asserted by the property test below
        p = readtable(df)
        @test !readtable(df; closed = true).stopfree
        @test readtable(df; closed = true).startfree
        @test !(p |> summarize(Count())).startfree
        @test !(p |> summarize(Count())).stopfree
        @test (p |> filterrows(r -> true) |> lag(1)).startfree
        @test !(clock(1) |> filterrows(r -> true)).startfree
        @test !asofjoin(p, p |> renamed(1)).startfree
        @test !CausalFrames.Acausal.futurejoin(p, p |> renamed(1)).stopfree
        @test !CausalPipeline(_ -> DataFrame[]).startfree
        # a frame's automatic `closed` rule holds only under the context check:
        # a run stopping past the frame isn't closed at the frame's stop
        f = load(
            Context(0, 10),
            readtable(DataFrame(time = [2, 10, 11], x = 1:3);
                closed = true),
        )
        @test readtable(f).stopfree
        @test !readtable(f; checkcontext = false).stopfree
        q = readtable(f; checkcontext = false)
        joined = DataFrame(
            load(Context(0, 10),
                CausalFrames.Acausal.futurejoin(q, q |> selectcolumns(:time, :x);
                    tolerance = 2, rightprefix = "r")),
        )
        @test joined.time == [2, 10]
    end

    @testset "a free pipeline's rows don't depend on its context" begin
        # The property the planner relies on: run over a wider context and
        # clipped, a startfree (stopfree) pipeline gives the rows of a run
        # over the narrower one.
        p = readtable(df)
        ticks = readtable(DataFrame(time = 0:3:45))
        table = DataFrame(k = ["a", "b"], w = [1.0, 2.0])
        free = (
            p |> filterrows(r -> r.x > 1) |> addcolumns(r -> (; y = 2r.x)),
            p |> selectcolumns(:time, :x) |> reordercolumns(:x),
            p |> lag(2),
            p |> CausalFrames.Acausal.lead(2),
            p |> fillmissing(:m => 0.0) |> sortcycles(:k; rev = true),
            p |> lookupjoin(table; key = :k),
            p |> summarizecycles([Count(), Sum(:x)]),
            p |> summarizecycles(Sum(:x); key = :k),
            p |> forwardfill(:m; key = :k, tolerance = 3),
            p |> warmup(3, forwardfill(:m; tolerance = 2)),
            summarizewindows(p, ticks, 4, [Count(), Mean(:x)]),
            summarizewindows(p, ticks, 4, Sum(:x); key = :k, keyset = ["a", "b"]),
            addrollingcolumns(p, (w3 = 3, w1 = 1), [Sum(:x), Max(:x)]; key = :k),
            asofjoin(p, p |> renamed(1) |> lag(1); key = :k, tolerance = 4),
            CausalFrames.Acausal.futurejoin(p, p |> renamed(1); key = :k,
                tolerance = 4),
            merge(p |> filterrows(r -> r.k == "a"), p |> dropcolumns(:m)),
            concatenate(p |> filterrows(r -> r.time < 20),
                p |> filterrows(r -> r.time >= 20)),
        )
        stopfree = (
            p |> head(7),
            clock(2),
            p |> addsummarycolumns(Sum(:x); key = :k),
            p |> settime(r -> r.time + 1),
            intervalize(p, ticks, Count()),
            asofjoin(p, p |> renamed(1) |> filterrows(r -> r.time % 10 == 5);
                key = :k),
            addrollingcolumns(p, (b3 = Bars(3),), Sum(:x)),
            p |> forwardfill(:m),
        )
        # dependent only through a key leaving the window, which this data
        # never does
        vanishing = summarizewindows(p, ticks, 4, Sum(:x); key = :k)
        loaded(c, q) = DataFrame(load(c, q))
        within(d, lo, hi) = d[lo .<= d.time .< hi, :]
        # Schemas are data-driven: an input with no rows in the narrow window
        # (a join's right side, a merge input) contributes no columns there,
        # and an empty load has only `:time`. So the wide run's rows must
        # equal the narrow run's on its columns, any others all `missing`.
        function same(wide, narrow)
            nrow(wide) == nrow(narrow) || return false
            nrow(wide) == 0 && return true
            extra = setdiff(names(wide), names(narrow))
            issubset(names(narrow), names(wide)) &&
            all(c -> all(ismissing, wide[!, c]), extra) || return false
            return isequal(wide[!, names(narrow)], narrow)
        end
        for (i, q) in enumerate(free)
            @test (i, q.startfree, q.stopfree) == (i, true, true)
        end
        for (i, q) in enumerate((stopfree..., vanishing))
            @test (i, q.startfree, q.stopfree) == (i, false, true)
        end
        for (a, b) in ((5, 30), (8, 21), (12, 13), (0, 40)), w in (1, 4, 9)
            for q in free
                @test same(within(loaded(Context(a - w, b), q), a, b),
                    loaded(Context(a, b), q))
            end
            for q in (free..., stopfree..., vanishing)
                @test same(within(loaded(Context(a, b + w), q), a, b),
                    loaded(Context(a, b), q))
            end
        end
        # and the dependent ones really do depend on their start
        for q in stopfree
            @test !same(within(loaded(Context(0, 30), q), 9, 30),
                loaded(Context(9, 30), q))
        end
    end

    @testset "shared and unshared runs agree on free pipelines" begin
        p = readtable(df)
        builds = (
            q -> asofjoin(q, q |> renamed(1) |> lag(3); key = :k, tolerance = 5),
            q -> addrollingcolumns(q, (w4 = 4, b3 = Bars(3)), [Sum(:x), Mean(:x)];
                key = :k),
            q -> merge(q |> filterrows(r -> r.k == "a"), q |> lag(1)),
            q -> summarizewindows(q, q |> filterrows(r -> r.k == "a"), 5,
                [Count(), Sum(:x)]; key = :k),
            q -> CausalFrames.Acausal.futurejoin(q, q |> renamed(2) |> lag(2);
                key = :k, tolerance = 3),
        )
        for f in builds, (a, b) in ((5, 35), (1, 2), (17, 40))
            c = Context(a, b)
            @test isequal(DataFrame(load(c, f(p))),
                DataFrame(load(c, f(p |> sharing(false)))))
            @test isequal(DataFrame(load(c, f(p))),
                DataFrame(load(c, f(p |> sharing(true)))))
            @test isequal(reduce(vcat, DataFrame.(stream(c, f(p))); init = DataFrame()),
                reduce(vcat, DataFrame.(stream(c, f(p |> sharing(false))));
                    init = DataFrame()))
        end
        # one row per chunk, so every clip falls inside a chunk boundary
        rows = CausalPipeline(; startfree = true, stopfree = true) do c
            (r for r in (df[i:i, :] for i in 1:nrow(df)) if c.start <= r.time[1] < c.stop)
        end
        for f in builds
            @test isequal(DataFrame(load(ctx, f(rows))), DataFrame(load(ctx, f(p))))
        end
    end

    @testset "sharing" begin
        p = readtable(df)
        @test isequal(DataFrame(load(ctx, p |> sharing(true))),
            DataFrame(load(ctx, p)))
        @test isequal(DataFrame(load(ctx, sharing(p, false))),
            DataFrame(load(ctx, p)))
        # the marker keeps its input's flags
        @test sharing(p, true).startfree
        @test !sharing(clock(1), true).startfree
        # a forced share of a dependent input shows its widened run
        cnt = clock(1) |> addsummarycolumns(Count())
        t = addrollingcolumns((w2 = 2,), Sum(:count))
        @test DataFrame(load(Context(3, 6), cnt |> t)).count == [1, 2, 3]
        @test DataFrame(load(Context(3, 6), cnt |> sharing(true) |> t)).count ==
              [3, 4, 5]
    end

    @testset "an abandoned side holds nothing back" begin
        src, runs = counting(df)
        chunked = CausalPipeline(; startfree = true, stopfree = true) do c
            (r for r in (df[i:(i+1), :] for i in 1:2:nrow(df))
                if c.start <= r.time[1] < c.stop)
        end
        for q in (chunked, src)
            p = merge(q |> head(2), q |> filterrows(r -> true))
            out = DataFrame(load(ctx, p))
            @test nrow(out) == 2 + nrow(df[5 .<= df.time .< 35, :])
            # dig out the shared instance of `q` and its tee
            slot = CausalFrames.planrun(p, ctx)
            it = CausalFrames.openslot!(slot)
            for _ in it
            end
            merged = slot.instance
            headslot = merged.inputs[1].instance.inputs[1]
            tee = headslot.instance.tee
            @test tee isa CausalFrames.ChunkTee
            @test headslot.abandoned
            @test all(isempty, tee.queues)
        end
        @test runs[] == 2   # one shared run per load above

        # an input `concatenate` never reaches never runs, so its own read of
        # the shared run is abandoned too, and queues nothing
        x =
            concatenate(chunked |> filterrows(r -> r.time < 20),
                chunked |> filterrows(r -> r.time >= 20)) |> head(1)
        p = merge(x, chunked)
        @test nrow(DataFrame(load(ctx, p))) == 1 + nrow(df[5 .<= df.time .< 35, :])
        slot = CausalFrames.planrun(p, ctx)
        for _ in CausalFrames.openslot!(slot)
        end
        tee = slot.instance.inputs[2].instance.tee
        @test length(tee.slots) == 3
        @test all(s -> s.abandoned, tee.slots[1:2])
        @test all(isempty, tee.queues)

        # head(0) never pulls; its side is abandoned from the start
        p = merge(chunked |> head(0), chunked)
        @test nrow(DataFrame(load(ctx, p))) == nrow(df[5 .<= df.time .< 35, :])

        # a reader stopping earlier than the run finishes its side early
        p = asofjoin(chunked, chunked |> renamed(1) |> lag(5); key = :k,
            tolerance = 10)
        q = chunked |> sharing(false)
        @test isequal(DataFrame(load(ctx, p)),
            DataFrame(
                load(ctx, asofjoin(q, q |> renamed(1) |> lag(5); key = :k,
                    tolerance = 10)),
            ))
    end
end
