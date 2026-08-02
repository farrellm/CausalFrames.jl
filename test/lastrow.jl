@testset "lastrow" begin
    ctx = Context(0, 10)
    src =
        clock(1; batchsize = 3) |>
        addcolumns(r -> (; k = isodd(r.time) ? "a" : "b", v = float(r.time)))

    @testset "keyless" begin
        df = DataFrame(load(ctx, src |> lastrow()))
        @test nrow(df) == 1
        @test names(df) == ["time", "k", "v"]
        @test df.time == [10]        # retimed to stop, not the row's own time 9
        @test df.k == ["a"] && df.v == [9.0]

        # :time keeps whatever position it had in the input
        offbeat = CausalPipeline() do _
            [DataFrame(a = [1, 2], time = [3, 4], b = ["x", "y"])]
        end
        odf = DataFrame(load(ctx, offbeat |> lastrow()))
        @test names(odf) == ["a", "time", "b"]
        @test odf.a == [2] && odf.time == [10] && odf.b == ["y"]

        # an empty input emits nothing at all, unlike keyless summarize, which
        # has an identity summary to fall back on
        blank = load(ctx, emptyframe() |> lastrow())
        @test nrow(blank) == 0 && names(blank) == ["time"]
        @test nrow(load(ctx, emptyframe() |> summarize(Count()))) == 1

        # the original timestamp is lost, but carrying it forward recovers it
        rdf = DataFrame(load(ctx, src |> addcolumns(r -> (; t0 = r.time)) |> lastrow()))
        @test rdf.t0 == [9] && rdf.time == [10]
    end

    @testset "keyed" begin
        df = DataFrame(load(ctx, src |> lastrow(; key = :k)))
        @test names(df) == ["time", "k", "v"]
        @test df.k == ["a", "b"]         # sorted by key
        @test df.time == [10, 10]        # all retimed to stop
        @test df.v == [9.0, 8.0]         # each key's own last row

        # keys last seen in different chunks, and a key seen only early on
        spread = CausalPipeline() do _
            [DataFrame(time = [1, 2], k = ["a", "c"], v = [1, 2]),
                DataFrame(time = [3, 4], k = ["b", "a"], v = [3, 4])]
        end
        sdf = DataFrame(load(ctx, spread |> lastrow(; key = :k)))
        @test sdf.k == ["a", "b", "c"]
        @test sdf.v == [4, 3, 2]
        @test sdf.time == [10, 10, 10]

        # a multi-column key sorts lexicographically on the key tuple
        multi = CausalPipeline() do _
            [
                DataFrame(time = [1, 2, 3, 4], g = ["y", "x", "x", "y"],
                    h = [1, 2, 1, 2], v = [10, 20, 30, 40]),
            ]
        end
        mdf = DataFrame(load(ctx, multi |> lastrow(; key = [:g, :h])))
        @test mdf.g == ["x", "x", "y", "y"]
        @test mdf.h == [1, 2, 1, 2]
        @test mdf.v == [30, 20, 10, 40]

        # an empty input emits no rows
        @test nrow(load(ctx, emptyframe() |> lastrow(; key = :k))) == 0
    end

    @testset "element types across chunks" begin
        # a source may hand a column a different element type per chunk; the
        # store tracks the promotion and widens a half-filled store in place
        drifting = CausalPipeline() do _
            [DataFrame(time = [1, 2], k = ["a", "b"], v = [1, 2]),
                DataFrame(time = [3], k = ["b"], v = [3.5])]
        end
        df = DataFrame(load(ctx, drifting |> lastrow(; key = :k)))
        @test df.k == ["a", "b"]
        @test df.v == [1.0, 3.5]
        @test eltype(df.v) == Float64      # "a" was stored as an Int and widened

        kdf = DataFrame(load(ctx, drifting |> lastrow()))
        @test kdf.v == [3.5] && eltype(kdf.v) == Float64
    end

    @testset "validation" begin
        # eager, at construction
        @test_throws ArgumentError lastrow(; key = :time)
        @test_throws ArgumentError lastrow(; key = [:a, :a])

        # the key column must be present, checked on the first chunk
        @test_throws ArgumentError load(ctx, src |> lastrow(; key = :nope))

        # the store's row type is fixed by the first chunk's names, so a chunk
        # that renames or reorders its columns is an error rather than an
        # opaque `convert` failure downstream
        renaming = CausalPipeline() do _
            [DataFrame(time = [1], k = ["a"], v = [1]),
                DataFrame(time = [2], k = ["a"], w = [2])]
        end
        @test_throws ArgumentError load(ctx, renaming |> lastrow(; key = :k))
        reordering = CausalPipeline() do _
            [DataFrame(time = [1], k = ["a"], v = [1]),
                DataFrame(time = [2], v = [2], k = ["a"])]
        end
        @test_throws ArgumentError load(ctx, reordering |> lastrow(; key = :k))
    end

    @testset "streaming and uncurried form" begin
        # like summarize, the stream is a single frame over the whole window
        frames = collect(stream(ctx, src |> lastrow(; key = :k)))
        @test length(frames) == 1
        @test context(only(frames)) == ctx
        @test isequal(DataFrame(only(frames)),
            DataFrame(load(ctx, src |> lastrow(; key = :k))))

        streamed = reduce(vcat, DataFrame.(stream(ctx, src |> lastrow())))
        @test isequal(streamed, DataFrame(load(ctx, src |> lastrow())))

        # the uncurried, pipeline-first form equals the |> chain
        @test isequal(DataFrame(load(ctx, lastrow(src; key = :k))),
            DataFrame(load(ctx, src |> lastrow(; key = :k))))
        @test isequal(DataFrame(load(ctx, lastrow(src))),
            DataFrame(load(ctx, src |> lastrow())))
    end
end
