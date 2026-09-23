@testset "readtable" begin
    df = DataFrame(time = [1, 2, 2, 5, 8], x = [1.0, 2.0, 3.0, 4.0, 5.0],
        s = ["a", "b", "c", "d", "e"])

    @testset "clipping, time conversion, column order" begin
        # the DataFrame path, and the generic path over columns and over rows
        for t in (df, Tables.columntable(copy(df)), Tables.rowtable(df))
            got = DataFrame(load(Context(2, 8), readtable(t)))
            @test got == df[2:4, :]
            @test names(got) == ["time", "x", "s"]
            @test DataFrame(load(Context(2, 8), readtable(t; closed = true))) ==
                  df[2:5, :]
            got = DataFrame(load(Context(0.0, 100.0), readtable(t)))
            @test eltype(got.time) == Float64
            @test got.time == [1.0, 2.0, 2.0, 5.0, 8.0]
            # a context wider than the table is fine: a table has no context
            @test nrow(load(Context(-100, 100), readtable(t))) == 5
        end
        for t in (DataFrame(time = Int[]), (time = Int[],))
            frame = load(Context(0, 10), readtable(t))
            @test nrow(frame) == 0
            @test names(frame) == ["time"]
        end
    end

    @testset "time selection" begin
        src = DataFrame(a = [1, 2, 3], ts = [10, 20, 30])
        for t in (src, Tables.columntable(src))
            got = DataFrame(load(Context(0, 25), readtable(t; time = :ts)))
            @test names(got) == ["a", "time"]
            @test got.time == [10, 20]
            got = DataFrame(load(Context(0, 25), readtable(t; time = r -> 10 * r.a)))
            @test names(got) == ["a", "ts", "time"]
            @test got.time == [10, 20]
        end
        # the DataFrame's own index is untouched by the rename
        @test names(src) == ["a", "ts"]
        # a function overwrites an existing :time where it stands
        for t in (df, Tables.columntable(df))
            got = DataFrame(load(Context(0, 100), readtable(t; time = r -> 10 * r.time)))
            @test names(got) == ["time", "x", "s"]
            @test got.time == [10, 20, 20, 50, 80]
        end

        ctx = Context(0, 100)
        textual = DataFrame(time = ["1", "2"])
        both = DataFrame(time = [1, 2], ts = [1, 2])
        # the DataFrame path reports where readtable was called ...
        @test_throws ArgumentError readtable(src)
        @test_throws ArgumentError readtable(src; time = :nope)
        @test_throws ArgumentError readtable(textual)
        @test_throws ArgumentError readtable(both; time = :ts)
        @test_throws ArgumentError readtable(src; time = r -> string(r.a))
        # ... the generic path when it runs
        @test_throws ArgumentError load(ctx, readtable(Tables.columntable(src)))
        @test_throws ArgumentError load(ctx,
            readtable(Tables.columntable(src); time = :nope))
        @test_throws ArgumentError load(ctx, readtable(Tables.columntable(textual)))
        @test_throws ArgumentError load(ctx,
            readtable(Tables.columntable(both); time = :ts))
        # a bad spec or a non-table is eager on every path
        @test_throws ArgumentError readtable(df; time = "time")
        @test_throws ArgumentError readtable(Tables.columntable(df); time = 1)
        @test_throws ArgumentError readtable(1)
        @test_throws ArgumentError readtable(clock(1))
    end

    @testset "order" begin
        ctx = Context(0, 10)
        unsorted = DataFrame(time = [3, 1, 2], x = [1, 2, 3])
        @test_throws ArgumentError readtable(unsorted)
        @test_throws ArgumentError load(ctx, readtable(Tables.columntable(unsorted)))
        @test readtable(unsorted; checkorder = false) isa CausalPipeline
        for t in (df, Tables.columntable(df))
            @test DataFrame(load(Context(0, 10), readtable(t; checkorder = false))) == df
        end

        # stable: rows sharing a timestamp keep their order
        ties = DataFrame(time = [2, 1, 2, 1], x = [1, 2, 3, 4])
        for t in (ties, Tables.columntable(ties))
            got = DataFrame(load(ctx, readtable(t; sort = true)))
            @test got.time == [1, 1, 2, 2]
            @test got.x == [2, 4, 1, 3]
            @test DataFrame(load(Context(2, 10), readtable(t; sort = true))).x == [1, 3]
            @test DataFrame(
                load(Context(0, 2), readtable(t; sort = true,
                    closed = true)),
            ).x == [2, 4, 1, 3]
        end
        @test ties.x == [1, 2, 3, 4]
    end

    @testset "partitioned tables" begin
        parts = Tables.partitioner([(time = [1, 2], x = [1.0, 2.0]),
            (time = [3, 4], x = [3.0, 4.0]), (time = [5, 6], x = [5.0, 6.0])])
        @test length(collect(stream(Context(0, 100), readtable(parts)))) == 3
        @test DataFrame(load(Context(2, 5), readtable(parts))).time == [2, 3, 4]
        # reading stops at the first partition past the window, so a later
        # out-of-order partition is only reported when it is reached
        bad = Tables.partitioner([(time = [1, 2],), (time = [3, 4],), (time = [0],)])
        @test DataFrame(load(Context(0, 3), readtable(bad))).time == [1, 2]
        @test_throws ArgumentError load(Context(0, 10), readtable(bad))
        # a sort concatenates the partitions
        @test DataFrame(load(Context(0, 10), readtable(bad; sort = true))).time ==
              [0, 1, 2, 3, 4]
    end

    @testset "missing times" begin
        ts = [missing, 1, 2, missing, 3, missing]
        src = DataFrame(time = ts, x = 1:6)
        for t in (src, Tables.columntable(src),
            Tables.partitioner([(time = ts[1:3], x = 1:3), (time = ts[4:6], x = 4:6)]))
            err = try
                load(Context(0, 10), readtable(t))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("skipmissing", sprint(showerror, err))
            for sort in (false, true)
                p = readtable(t; skipmissing = true, sort)
                df = DataFrame(load(Context(0, 10), p))
                @test df.time == [1, 2, 3]
                @test df.x == [2, 3, 5]
                @test eltype(df.time) == Int
                @test DataFrame(load(Context(2, 3), p)).x == [3]
                @test DataFrame(
                    load(Context(2, 3),
                        readtable(t; skipmissing = true, closed = true)),
                ).x == [3, 5]
            end
        end
        # a time function returning missing, and a DataFrame left untouched
        p = readtable(src; skipmissing = true,
            time = r -> ismissing(r.time) ? missing : 10 * r.time)
        @test DataFrame(load(Context(0, 100), p)).time == [10, 20, 30]
        @test isequal(src.time, ts)
        @test names(src) == ["time", "x"]
        # the rows are private copies, so whole-window runs may share them
        p = readtable(src; skipmissing = true)
        @test only(load(Context(0, 10), p).chunks).x ===
              only(load(Context(0, 10), p).chunks).x
        # a column admitting missing with none in it needs no option
        @test DataFrame(
            load(Context(0, 10),
                readtable((time = Union{Missing,Int}[1, 2], x = [1, 2]))),
        ).time == [1, 2]
        # an all-missing column is not text
        @test nrow(
            load(Context(0, 10),
                readtable((time = [missing, missing], x = [1, 2]); skipmissing = true)),
        ) == 0
    end

    @testset "rows are copied, never aliased" begin
        src = DataFrame(time = [1, 2, 3], x = [1.0, 2.0, 3.0])
        for t in (src, Tables.columntable(src))
            frame = load(Context(0, 10), readtable(t))
            @test only(frame.chunks).x !== src.x
            @test only(frame.chunks).time !== src.time
        end
        frame = load(Context(0, 10), readtable(src))
        src.x[1] = 100.0
        @test DataFrame(frame).x == [1.0, 2.0, 3.0]
        # a sort that permuted made copies of its own, so whole-window runs
        # share those
        rev = DataFrame(time = [3, 2, 1], x = [1.0, 2.0, 3.0])
        p = readtable(rev; sort = true)
        a = load(Context(0, 10), p)
        b = load(Context(0, 10), p)
        @test only(a.chunks).x === only(b.chunks).x
        @test only(a.chunks).x !== rev.x
        @test rev.x == [1.0, 2.0, 3.0]
    end

    @testset "the DataFrame path agrees with the generic path" begin
        src = DataFrame(ts = [5, 1, 3, 3, 9, 7], k = [1, 2, 3, 4, 5, 6],
            v = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        generic = Tables.columntable(src)
        kwsets = [(; time = :ts, sort = true), (; time = :ts, sort = true, closed = true),
            (; time = :k), (; time = r -> r.k, closed = true),
            (; time = r -> 2 * r.ts, sort = true)]
        ctxs = [Context(0, 100), Context(2, 9), Context(3, 7), Context(9, 9),
            Context(0.0, 6.0)]
        for kw in kwsets, ctx in ctxs
            want = DataFrame(load(ctx, readtable(generic; kw...)))
            got = DataFrame(load(ctx, readtable(src; kw...)))
            @test got == want
            @test eltype.(eachcol(got)) == eltype.(eachcol(want))
        end
    end

    @testset "streaming concatenates to load" begin
        ctx = Context(0, 10)
        parts = Tables.partitioner([(time = [1, 2], x = [1, 2]), (time = [4], x = [3])])
        for p in (readtable(df), readtable(Tables.columntable(df)), readtable(parts),
            readtable(load(ctx, readtable(parts))))
            @test reduce(vcat, DataFrame.(stream(ctx, p))) == DataFrame(load(ctx, p))
        end
    end

    @testset "CausalFrame" begin
        ctx = Context(0, 10)
        src = CausalPipeline(
            ctx -> [DataFrame(time = [0, 2, 4], x = [1, 2, 3]),
                DataFrame(time = [6, 8], x = [4, 5])],
        )
        frame = load(ctx, src)

        # a round trip keeps the rows and the chunk structure
        @test DataFrame(load(ctx, readtable(frame))) == DataFrame(frame)
        @test length(collect(stream(ctx, readtable(frame)))) == 2
        @test nrow(load(ctx, readtable(load(ctx, emptyframe())))) == 0

        # rows at the frame's stop are kept exactly when the stops match
        summary = load(ctx, src |> summarize(Sum(:x)))
        @test DataFrame(load(ctx, readtable(summary))) == DataFrame(summary)
        @test nrow(load(ctx, readtable(summary; closed = false))) == 0
        @test nrow(load(Context(0, 20), readtable(summary; checkcontext = false))) == 1
        @test nrow(load(Context(0, 5), readtable(summary))) == 0

        # a narrower window is half-open unless closed
        @test DataFrame(load(Context(2, 8), readtable(frame))).time == [2, 4, 6]
        @test DataFrame(load(Context(2, 8), readtable(frame; closed = true))).time ==
              [2, 4, 6, 8]
        @test nrow(load(Context(5, 5), readtable(frame))) == 0
        @test nrow(load(Context(6, 6), readtable(frame; closed = true))) == 1
        got = DataFrame(load(Context(0.0, 10.0), readtable(frame)))
        @test eltype(got.time) == Float64
        @test got.time == [0.0, 2.0, 4.0, 6.0, 8.0]

        # outside the frame's context the rows are unknown
        @test_throws ArgumentError load(Context(-1, 10), readtable(frame))
        @test_throws ArgumentError load(Context(0, 11), readtable(frame))
        @test DataFrame(load(Context(-5, 7), readtable(frame; checkcontext = false))).time ==
              [0, 2, 4, 6]

        # the table-only keywords have nothing to act on
        @test_throws MethodError readtable(frame; sort = true)
        @test_throws MethodError readtable(frame; time = :x)
        @test_throws MethodError readtable(frame; checkorder = false)

        # a chunk wholly inside the window shares its vectors; a partial one is
        # sliced
        whole = load(ctx, readtable(frame))
        @test whole.chunks[1].x === frame.chunks[1].x
        part = load(Context(2, 10), readtable(frame))
        @test part.chunks[1].x !== frame.chunks[1].x
        @test part.chunks[2].x === frame.chunks[2].x

        # a downstream index mutation (asofjoin prefixes the left chunk in
        # place) cannot reach the source frame
        joined = load(
            ctx,
            readtable(frame) |>
            asofjoin(readtable(frame); leftprefix = "l", rightprefix = "r"),
        )
        @test nrow(joined) == 5
        @test names(frame) == ["time", "x"]
        @test all(c -> names(c) == ["time", "x"], frame.chunks)
        @test DataFrame(frame).x == [1, 2, 3, 4, 5]
    end
end
