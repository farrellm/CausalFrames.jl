@testset "fill" begin
    ctx = Context(0, 10)
    onechunk(; cols...) = CausalPipeline(_ -> [DataFrame(; cols...)])

    @testset "fillmissing" begin
        p = onechunk(time = [1, 2, 3], q = [1.0, missing, 3.0],
            s = ["a", missing, "c"])
        df = DataFrame(load(ctx, p |> fillmissing(:q => 0.0, :s => "")))
        @test df.q == [1.0, 0.0, 3.0]
        @test df.s == ["a", "", "c"]
        # the Missing is gone from the element type, since every one is filled
        @test eltype(df.q) == Float64
        @test eltype(df.s) == String

        # the three spec spellings agree
        nt = DataFrame(load(ctx, p |> fillmissing((q = 0.0, s = ""))))
        coll = DataFrame(load(ctx, p |> fillmissing([:q => 0.0, :s => ""])))
        @test nt == df && coll == df

        # the fill value promotes with the column's non-missing type
        w = DataFrame(
            load(ctx, onechunk(time = [1, 2], v = [1, missing]) |>
                fillmissing(:v => 0.5)),
        )
        @test w.v == [1.0, 0.5] && eltype(w.v) == Float64

        # a column that admits no Missing is left untouched
        u = onechunk(time = [1, 2], v = [1, 2])
        @test DataFrame(load(ctx, u |> fillmissing(:v => 9))).v == [1, 2]

        # an all-Missing column takes the fill value's own type
        a = onechunk(time = [1, 2], v = [missing, missing])
        adf = DataFrame(load(ctx, a |> fillmissing(:v => 7)))
        @test adf.v == [7, 7] && eltype(adf.v) == Int

        # eager, at construction
        @test_throws ArgumentError fillmissing()
        @test_throws ArgumentError fillmissing(:a => 1, :a => 2)
        @test_throws ArgumentError fillmissing(:time => 0)
        @test_throws ArgumentError fillmissing(:a)
        # and when the chunk arrives, for a column the data does not have
        @test_throws ArgumentError load(ctx, p |> fillmissing(:nope => 0))
    end

    @testset "forwardfill keyless" begin
        # the gap spans the chunk boundary, so the carry is across chunks
        src = CausalPipeline() do _
            [DataFrame(time = [1, 2, 3], v = [missing, 2.0, missing]),
                DataFrame(time = [4, 5], v = [missing, 5.0])]
        end
        df = DataFrame(load(ctx, src |> forwardfill(:v)))
        @test isequal(df.v, [missing, 2.0, 2.0, 2.0, 5.0])
        # leading rows have nothing to carry, so the type keeps its Missing
        @test eltype(df.v) == Union{Missing,Float64}
        @test names(df) == ["time", "v"]

        # each column carries independently: staggered gaps do not sync up
        two = onechunk(time = [1, 2, 3, 4],
            a = [1, missing, missing, missing],
            b = [missing, missing, 3, missing])
        tdf = DataFrame(load(ctx, two |> forwardfill(:a, :b)))
        @test isequal(tdf.a, [1, 1, 1, 1])
        @test isequal(tdf.b, [missing, missing, 3, 3])

        # a selected column admitting no Missing is passed through untouched
        clean = onechunk(time = [1, 2], v = [1, 2])
        cdf = DataFrame(load(ctx, clean |> forwardfill(:v)))
        @test cdf.v == [1, 2] && eltype(cdf.v) == Int
    end

    @testset "forwardfill keyed" begin
        src = CausalPipeline() do _
            [
                DataFrame(time = [1, 2, 3], k = ["a", "b", "a"],
                    v = [1.0, missing, missing]),
                DataFrame(time = [4, 5], k = ["b", "a"], v = [missing, missing])]
        end
        df = DataFrame(load(ctx, src |> forwardfill(:v; key = :k)))
        # "b" never had a value of its own, so "a"'s never leaks into its rows
        @test isequal(df.v, [1.0, missing, 1.0, missing, 1.0])

        # a multi-column key
        multi = onechunk(time = [1, 2, 3, 4], g = ["x", "x", "y", "x"],
            h = [1, 2, 1, 1], v = [10, missing, missing, missing])
        mdf = DataFrame(load(ctx, multi |> forwardfill(:v; key = [:g, :h])))
        @test isequal(mdf.v, [10, missing, missing, 10])

        # the key columns are never filled, even when a selector matches them
        kdf = DataFrame(
            load(ctx,
                onechunk(time = [1, 2], k = ["a", "a"], v = [1, missing]) |>
                forwardfill(n -> true; key = :k)),
        )
        @test names(kdf) == ["time", "k", "v"]
        @test kdf.k == ["a", "a"] && isequal(kdf.v, [1, 1])
    end

    @testset "tolerance" begin
        src = onechunk(time = [1, 3, 5], v = [1.0, missing, missing])
        # inclusive boundary: distance == tolerance still carries
        df = DataFrame(load(ctx, src |> forwardfill(:v; tolerance = 2)))
        @test isequal(df.v, [1.0, 1.0, missing])   # 5 - 1 = 4 > 2: gone stale

        # tolerance = 0 carries into rows sharing the value's own time only
        same = onechunk(time = [1, 1, 2], v = [1.0, missing, missing])
        zdf = DataFrame(load(ctx, same |> forwardfill(:v; tolerance = 0)))
        @test isequal(zdf.v, [1.0, 1.0, missing])

        # the input context is widened by the tolerance...
        seen = Ref{Any}(nothing)
        recorder = CausalPipeline() do c
            seen[] = c
            [DataFrame(time = [c.start, 3], v = [1.0, missing])]
        end
        wdf = DataFrame(load(Context(3, 10), recorder |> forwardfill(:v;
            tolerance = 2)))
        @test seen[] == Context(1, 10)
        # ...the pre-window row filled the cell and was then dropped
        @test wdf.time == [3] && isequal(wdf.v, [1.0])

        # ...and untouched without one
        load(Context(3, 10), recorder |> forwardfill(:v))
        @test seen[] == Context(3, 10)

        # a chunk lying entirely before the window is dropped whole
        split = CausalPipeline() do c
            [DataFrame(time = [c.start], v = [1.0]),
                DataFrame(time = [3], v = [missing])]
        end
        sdf = DataFrame(load(Context(3, 10), split |> forwardfill(:v;
            tolerance = 2)))
        @test sdf.time == [3] && isequal(sdf.v, [1.0])

        # negative tolerance is rejected when the pipeline runs
        @test_throws ArgumentError load(ctx, src |> forwardfill(:v; tolerance = -1))
    end

    @testset "selectors" begin
        p = onechunk(time = [1, 2], px_a = [1.0, missing], px_b = [2.0, missing],
            other = [3.0, missing])
        rdf = DataFrame(load(ctx, p |> forwardfill(r"^px_")))
        @test isequal(rdf.px_a, [1.0, 1.0]) && isequal(rdf.px_b, [2.0, 2.0])
        @test isequal(rdf.other, [3.0, missing])   # unselected, so untouched

        fdf = DataFrame(load(ctx, p |> forwardfill(n -> endswith(n, "_b"))))
        @test isequal(fdf.px_a, [1.0, missing]) && isequal(fdf.px_b, [2.0, 2.0])

        # a selector matching nothing is a no-op, but a named column the data
        # does not have is an error
        @test isequal(DataFrame(load(ctx, p |> forwardfill(r"^zz"))).other,
            [3.0, missing])
        @test_throws ArgumentError load(ctx, p |> forwardfill(:nope))

        # :time is never filled, even when a selector matches it
        tdf = DataFrame(load(ctx, p |> forwardfill(:time, :other)))
        @test tdf.time == [1, 2] && isequal(tdf.other, [3.0, 3.0])
    end

    @testset "element types across chunks" begin
        # a source may hand a column a different element type per chunk; the
        # cells track the promotion and widen in place
        drifting = CausalPipeline() do _
            [DataFrame(time = [1, 2], k = ["a", "b"], v = [1, missing]),
                DataFrame(time = [3, 4], k = ["b", "a"], v = [3.5, missing])]
        end
        df = DataFrame(load(ctx, drifting |> forwardfill(:v; key = :k)))
        @test isequal(df.v, [1, missing, 3.5, 1.0])
        @test eltype(df.v) == Union{Missing,Float64}   # "a" was stored as an Int

        kdf = DataFrame(load(ctx, drifting |> forwardfill(:v)))
        @test isequal(kdf.v, [1, 1, 3.5, 3.5])
    end

    @testset "validation" begin
        # eager, at construction
        @test_throws ArgumentError forwardfill()
        @test_throws ArgumentError forwardfill(:v; key = :time)
        @test_throws ArgumentError forwardfill(:v; key = [:a, :a])
        @test_throws ArgumentError forwardfill(1)

        src = onechunk(time = [1], v = [1])
        # the key column must be present, checked on the first chunk
        @test_throws ArgumentError load(ctx, src |> forwardfill(:v; key = :nope))

        # the cells' names are fixed by the columns first resolved, so a chunk
        # that changes the set being filled is an error rather than an opaque
        # `convert` failure
        shifting = CausalPipeline() do _
            [DataFrame(time = [1], px_a = [1]),
                DataFrame(time = [2], px_a = [2], px_b = [3])]
        end
        @test_throws ArgumentError load(ctx, shifting |> forwardfill(r"^px_"))
    end

    @testset "streaming and uncurried form" begin
        src =
            clock(1; batchsize = 3) |>
            addcolumns(
                r -> (; k = isodd(r.time) ? "a" : "b",
                    v = r.time % 3 == 0 ? missing : float(r.time)),
            )

        for p in (src |> forwardfill(:v),
            src |> forwardfill(:v; key = :k),
            src |> forwardfill(:v; key = :k, tolerance = 3),
            src |> fillmissing(:v => 0.0))
            streamed = reduce(vcat, DataFrame.(stream(ctx, p)))
            @test isequal(streamed, DataFrame(load(ctx, p)))
        end

        # the uncurried, pipeline-first form equals the |> chain
        @test isequal(DataFrame(load(ctx, forwardfill(src, :v; key = :k))),
            DataFrame(load(ctx, src |> forwardfill(:v; key = :k))))
        @test isequal(DataFrame(load(ctx, fillmissing(src, :v => 0.0))),
            DataFrame(load(ctx, src |> fillmissing(:v => 0.0))))
    end

    @testset "kernel allocates nothing per row" begin
        # The cells are mutable and live in a NamedTuple, precisely so that a
        # carried value that is not isbits — a String — costs no box per row.
        # Nothing else in the suite would notice if that regressed.
        function fillalloc(n = 200)
            C = CausalFrames.FillCell{String,Int}
            times = collect(1:n)
            incol = Vector{Union{Missing,String}}(undef, n)
            for i in 1:n
                incol[i] = iseven(i) ? missing : "s" * string(i % 5)
            end
            outcol = Vector{Union{Missing,String}}(undef, n)
            groups = ((incol, outcol),)
            cells = (C(),)
            call() = CausalFrames.fillkeyless!(times, groups, cells, 2)
            call()                      # warm up: the cell gains a value
            return @allocated call()
        end
        @test fillalloc() == 0
    end
end
