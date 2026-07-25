@testset "merge" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])
    ctx = Context(0, 10)

    @testset "basic" begin
        a = onechunk(time = [0, 2, 4], x = [1, 2, 3])
        b = onechunk(time = [1, 2, 5], y = ["a", "b", "c"])
        df = DataFrame(load(ctx, merge(a, b)))

        # rows interleave by time; :time leads the union schema, then the
        # columns in the order the pipelines introduce them
        @test names(df) == ["time", "x", "y"]
        @test df.time == [0, 1, 2, 2, 4, 5]
        @test isequal(df.x, [1, missing, 2, missing, 3, missing])
        @test isequal(df.y, [missing, "a", missing, "b", missing, "c"])

        # a column absent from a pipeline is missing for its rows, so the
        # merged column widens
        @test eltype(df.x) == Union{Missing,Int}
        @test eltype(df.y) == Union{Missing,String}
        @test eltype(df.time) == Int

        # pipelines disjoint in time merge to their concatenation
        early = clock(1) |> filterrows(r -> r.time < 4)
        late = clock(1) |> filterrows(r -> r.time >= 4)
        @test isequal(DataFrame(load(ctx, merge(early, late))),
            DataFrame(load(ctx, concatenate(early, late))))
    end

    @testset "ties" begin
        # rows at equal times are emitted in argument order, in runs
        a = onechunk(time = [5, 5, 5], src = ["a", "a", "a"], i = [1, 2, 3])
        b = onechunk(time = [5, 5], src = ["b", "b"], i = [1, 2])
        c = onechunk(time = [5], src = ["c"], i = [1])
        df = DataFrame(load(ctx, merge(a, b, c)))
        @test df.src == ["a", "a", "a", "b", "b", "c"]
        @test df.i == [1, 2, 3, 1, 2, 1]

        # and in argument order at every tie, not just the first
        x = onechunk(time = [1, 3], src = ["x", "x"])
        y = onechunk(time = [1, 2, 3], src = ["y", "y", "y"])
        @test DataFrame(load(ctx, merge(y, x))).src ==
              ["y", "x", "y", "y", "x"]

        # each pipeline's own row order is preserved within a tie
        @test DataFrame(load(ctx, merge(a, a))).i == [1, 2, 3, 1, 2, 3]
    end

    @testset "schema" begin
        # :time leads however the pipeline orders its columns
        p = onechunk(a = [1], time = [1], b = [2])
        q = onechunk(c = [3], time = [2])
        @test names(DataFrame(load(ctx, merge(p, q)))) == ["time", "a", "b", "c"]

        # one pipeline is that pipeline
        whole = clock(1) |> addcolumns(r -> (; v = float(r.time)))
        @test isequal(DataFrame(load(ctx, merge(whole))),
            DataFrame(load(ctx, whole)))

        # a pipeline producing no chunks contributes no columns
        @test isequal(DataFrame(load(ctx, merge(whole, emptyframe()))),
            DataFrame(load(ctx, whole)))
        @test isequal(DataFrame(load(ctx, merge(emptyframe(), whole))),
            DataFrame(load(ctx, whole)))

        # all of them empty is a zero-row frame with only :time
        blank = load(ctx, merge(emptyframe(), emptyframe()))
        @test nrow(blank) == 0
        @test names(blank) == ["time"]

        # element types promote across pipelines, as they do across chunks
        ints = onechunk(time = [1], x = [1])
        floats = onechunk(time = [2], x = [2.5])
        @test DataFrame(load(ctx, merge(ints, floats))).x == [1.0, 2.5]
        @test eltype(DataFrame(load(ctx, merge(ints, floats))).x) == Float64

        # so do time columns of differing types
        ftime = onechunk(time = [1.5], x = [1])
        @test DataFrame(load(Context(0.0, 10.0), merge(ints, ftime))).time ==
              [1.0, 1.5]

        # a pipeline whose columns change mid-stream is rejected, whether
        # renamed or merely reordered
        renamed = CausalPipeline(
            c -> [DataFrame(time = [1], a = [1]),
                DataFrame(time = [2], b = [2])],
        )
        reordered = CausalPipeline(
            c -> [DataFrame(time = [1], a = [1]),
                DataFrame(a = [2], time = [2])],
        )
        @test_throws ArgumentError load(ctx, merge(renamed, emptyframe()))
        @test_throws ArgumentError load(ctx, merge(reordered, emptyframe()))

        # a pipeline without :time cannot be ordered
        @test_throws ArgumentError load(ctx, merge(onechunk(x = [1])))
    end

    @testset "arity and batchsize" begin
        # there is no zero-argument form — emptyframe is the identity
        @test_throws MethodError merge()

        # batchsize is validated before the pipeline is ever run
        @test_throws ArgumentError merge(emptyframe(); batchsize = 0)
        @test_throws ArgumentError merge(emptyframe(); batchsize = -1)

        # blocks are batched: a row-by-row interleave still yields chunks of
        # at least batchsize rows, the tail excepted, and never more than one
        # input chunk beyond it
        a = CausalPipeline(c -> [DataFrame(time = collect(0:2:98), x = 1:50)])
        b = CausalPipeline(c -> [DataFrame(time = collect(1:2:99), x = 1:50)])
        big = Context(0, 100)
        sizes = [nrow(c) for c in merge(a, b; batchsize = 8).run(big)]
        @test sum(sizes) == 100
        @test all(>=(8), sizes[1:(end-1)])
        @test all(<=(8 + 50 - 1), sizes)

        # batchsize 1 emits one chunk per block, so an alternating merge
        # yields one row at a time
        @test [nrow(c) for c in merge(a, b; batchsize = 1).run(big)] ==
              fill(1, 100)

        # batching does not change the rows
        @test isequal(DataFrame(load(big, merge(a, b; batchsize = 8))),
            DataFrame(load(big, merge(a, b; batchsize = 1))))
    end

    @testset "chunks and laziness" begin
        # every pipeline is run over the full context, and clips itself
        seen = Context[]
        recorder(t) = CausalPipeline() do c
            push!(seen, c)
            return [DataFrame(time = [t], x = [t])]
        end
        load(ctx, merge(recorder(1), recorder(2)))
        @test seen == [ctx, ctx]

        # pulls are counted per chunk: one from each pipeline before the first
        # output chunk, and never more than one chunk of lookahead after
        pulls = [0, 0]
        counted(i, dfs) = CausalPipeline(c -> (
            begin
                pulls[i] += 1
                d
            end for d in dfs
        ))
        a = counted(1, [DataFrame(time = [0], x = [1]),
            DataFrame(time = [4], x = [2])])
        b = counted(2, [DataFrame(time = [2], x = [3]),
            DataFrame(time = [6], x = [4])])
        it = merge(a, b; batchsize = 1).run(ctx)
        next = iterate(it)
        @test pulls == [1, 1]
        @test next[1].time == [0]
        next = iterate(it, next[2])
        # pipeline 1's chunk is spent, so it refills; pipeline 2's is not
        @test pulls == [2, 1]
        @test next[1].time == [2]

        # chunk boundaries do not disturb the interleaving
        split = CausalPipeline(
            c -> [DataFrame(time = [0, 3], x = [1, 2]),
                DataFrame(time = [5], x = [3]),
                DataFrame(time = [7, 9], x = [4, 5])],
        )
        other = CausalPipeline(
            c -> [DataFrame(time = [1, 4], y = [1, 2]),
                DataFrame(time = [6, 8], y = [3, 4])],
        )
        merged = merge(split, other)
        df = DataFrame(load(ctx, merged))
        @test df.time == [0, 1, 3, 4, 5, 6, 7, 8, 9]
        @test isequal(df.x, [1, missing, 2, missing, 3, missing, 4, missing, 5])
        @test isequal(df.y, [missing, 1, missing, 2, missing, 3, missing, 4,
            missing])

        # empty chunks mid-stream are skipped
        gappy = CausalPipeline(
            c -> [DataFrame(time = [0, 3], x = [1, 2]),
                DataFrame(time = Int[], x = Int[]),
                DataFrame(time = [5], x = [3]),
                DataFrame(time = [7, 9], x = [4, 5])],
        )
        @test isequal(DataFrame(load(ctx, merge(gappy, other))), df)

        # streaming matches load
        streamed = reduce(vcat,
            [DataFrame(f) for f in stream(ctx, merge(split, other; batchsize = 2))])
        @test isequal(streamed, df)
    end

    @testset "whole chunks are not copied" begin
        # a pipeline whose columns are already the union schema hands its
        # chunks through by reference when nothing interleaves into them
        chunk = DataFrame(time = [0, 1, 2], x = [1, 2, 3])
        a = CausalPipeline(c -> [chunk])
        b = onechunk(time = [9], x = [4])
        out = first(iterate(merge(a, b; batchsize = 1).run(ctx)))
        @test out === chunk
        @test out.x === chunk.x
    end

    @testset "chains" begin
        a = onechunk(time = [0, 2, 4], x = [1.0, 2.0, 3.0])
        b = onechunk(time = [1, 3], y = [10.0, 20.0])

        # downstream
        df = DataFrame(load(ctx, merge(a, b) |> filterrows(r -> r.time > 1)))
        @test df.time == [2, 3, 4]
        df = DataFrame(load(ctx, merge(a, b) |> summarize(Count())))
        @test df.count == [5]
        df = DataFrame(load(ctx, merge(a, b) |> addsummarycolumns(Count())))
        @test df.count == 1:5

        # and upstream — the inputs may be transforms themselves
        shifted = merge(a |> lag(1), b)
        @test DataFrame(load(ctx, shifted)).time == [1, 1, 3, 3, 5]

        # a non-numeric time type merges like any other
        t0 = DateTime(2024, 1, 1)
        dts = Context(t0, t0 + Day(1))
        p = onechunk(time = [t0, t0 + Hour(2)], x = [1, 2])
        q = onechunk(time = [t0 + Hour(1)], y = [3])
        @test DataFrame(load(dts, merge(p, q))).time ==
              [t0, t0 + Hour(1), t0 + Hour(2)]
    end

    @testset "differential" begin
        # against a naive oracle: tag every row with its pipeline's argument
        # index and its position, then sort by (time, index, position)
        vals = lcgsequence(20240724, 4000, 100)
        pos = 1
        take!(n) = (v = vals[pos:(pos+n-1)]; pos += n; v)
        for trial in 1:20
            npipes = 2 + trial % 4
            frames = DataFrame[]
            pipes = CausalPipeline[]
            for i in 1:npipes
                nrows = 1 + take!(1)[1] % 12
                times = sort(take!(nrows) .% 40)
                # a random subset of the columns, so schemas differ
                cols = [(:a, 1.0), (:b, 2), (:c, 3)][1:(1+i%3)]
                df = DataFrame(:time => times,
                    (n => [v * (1 + j) for j in 1:nrows] for (n, v) in cols)...)
                push!(frames, df)
                # split into chunks at random boundaries
                cuts = sort(unique(take!(2) .% nrows))
                bounds = unique([0; cuts; nrows])
                chunks = [df[(bounds[k]+1):bounds[k+1], :]
                          for k in 1:(length(bounds)-1)]
                push!(pipes, CausalPipeline(c -> chunks))
            end

            wide = Context(0, 40)
            got = DataFrame(load(wide, merge(pipes...; batchsize = 3)))

            allnames = unique(reduce(vcat, [propertynames(f) for f in frames]))
            tagged = DataFrame[]
            for (i, f) in enumerate(frames)
                g = copy(f)
                for n in allnames
                    n in propertynames(g) || (g[!, n] = fill(missing, nrow(g)))
                end
                g[!, :src] .= i
                g[!, :ord] = 1:nrow(g)
                push!(tagged, g[!, [allnames..., :src, :ord]])
            end
            want = sort(reduce(vcat, tagged), [:time, :src, :ord])
            select!(want, allnames)

            @test names(got) == names(want)
            @test isequal(got, want)
        end
    end
end
