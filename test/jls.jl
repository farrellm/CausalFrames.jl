@testset "writejls and readjls" begin
    dir = mktempdir()
    ctx = Context(0, 100)
    # Columns neither CSV nor parquet can carry — vectors and NamedTuples —
    # over two chunks, built fresh per run since chunks are consumed by
    # ownership.
    p = CausalPipeline(
        ctx -> [
            DataFrame(time = [1, 3], x = [1.5, 2.5], v = [[1, 2], [3]],
                nt = [(a = 1, b = "x"), (a = 2, b = "y")]),
            DataFrame(time = [5, 7], x = [3.5, 4.5], v = [Int[], [4, 5, 6]],
                nt = [(a = 3, b = "z"), (a = 4, b = "w")]),
        ],
    )
    want = DataFrame(load(ctx, p))

    @testset "round trip" begin
        out = joinpath(dir, "out.jls")
        # pass-through: the frame is unchanged
        @test DataFrame(load(ctx, p |> writejls(out))) == want
        back = DataFrame(load(ctx, readjls(out)))
        @test back == want
        @test eltype(back.v) == Vector{Int}
        @test eltype(back.nt) == eltype(want.nt)
        # one chunk per record
        @test length(collect(stream(ctx, readjls(out)))) == 2

        # uncurried form, scan, and queue = 0 all write the same file
        out2 = joinpath(dir, "out2.jls")
        scan(ctx, writejls(p, out2))
        @test read(out2) == read(out)
        rendez = joinpath(dir, "rendez.jls")
        scan(ctx, p |> writejls(rendez; queue = 0))
        @test read(rendez) == read(out)

        # re-running truncates rather than appending
        scan(ctx, p |> writejls(out))
        @test read(out) == read(out2)
    end

    @testset "window clipping and time conversion" begin
        out = joinpath(dir, "clip.jls")
        scan(ctx, p |> writejls(out))
        df = DataFrame(load(Context(2, 6), readjls(out)))
        @test df.time == [3, 5]
        @test df.x == [2.5, 3.5]
        # the time column takes the context's time type
        df = DataFrame(load(Context(0.0, 100.0), readjls(out)))
        @test eltype(df.time) == Float64
        @test df.time == [1.0, 3.0, 5.0, 7.0]

        ticks = joinpath(dir, "ticks.jls")
        dctx = Context(DateTime(2026, 1, 1), DateTime(2026, 1, 1, 1))
        scan(dctx, clock(Minute(10); batchsize = 2) |> writejls(ticks))
        @test DataFrame(load(dctx, readjls(ticks))) ==
              DataFrame(load(dctx, clock(Minute(10))))
    end

    @testset "an empty stream leaves a header-only file" begin
        none = joinpath(dir, "none.jls")
        scan(ctx, emptyframe() |> writejls(none))
        @test isfile(none) && filesize(none) > 0
        df = DataFrame(load(ctx, readjls(none)))
        @test nrow(df) == 0
        @test names(df) == ["time"]
    end

    @testset "ownership against downstream index mutation" begin
        # the writer serializes its chunk on another task while asofjoin's
        # leftprefix renames the downstream chunk's columns in place
        q = CausalPipeline(ctx -> [DataFrame(time = [1, 2], a = [1, 2])])
        mid = joinpath(dir, "mid.jls")
        frame = load(ctx,
            q |> writejls(mid) |> asofjoin(q; leftprefix = "l", rightprefix = "r"))
        @test names(frame) == ["time", "l_a", "r_a"]
        @test names(DataFrame(load(ctx, readjls(mid)))) == ["time", "a"]
    end

    @testset "errors" begin
        @test_throws ArgumentError writejls(joinpath(dir, "x.jls"); queue = -1)

        # chunk columns may not change mid-stream
        shifty = CausalPipeline(
            ctx -> [DataFrame(time = [1], a = [1]), DataFrame(time = [2], b = [2])])
        @test_throws ArgumentError scan(ctx, shifty |> writejls(joinpath(dir, "s.jls")))

        # a writer failure propagates to the consumer
        bad = joinpath(dir, "nosuchdir", "out.jls")
        @test_throws Exception scan(ctx, p |> writejls(bad))

        # files writejls did not write: empty, text, another value, a future
        # format version
        empty = joinpath(dir, "empty.jls")
        touch(empty)
        @test_throws ArgumentError load(ctx, readjls(empty))
        text = joinpath(dir, "text.jls")
        write(text, "time,x\n1,2\n")
        @test_throws ArgumentError load(ctx, readjls(text))
        other = joinpath(dir, "other.jls")
        serialize(other, 42)
        @test_throws ArgumentError load(ctx, readjls(other))
        future = joinpath(dir, "future.jls")
        open(io -> serialize(io, (format = :CausalFramesJLS, version = 99)),
            future, "w")
        @test_throws ArgumentError load(ctx, readjls(future))

        # a record that is not a chunk
        notchunk = joinpath(dir, "notchunk.jls")
        open(notchunk, "w") do io
            serialize(io, CausalFrames.JLSHEADER)
            serialize(io, [1, 2, 3])
        end
        @test_throws ArgumentError load(ctx, readjls(notchunk))

        # records out of time order are caught across the record boundary
        unsorted = joinpath(dir, "unsorted.jls")
        open(unsorted, "w") do io
            serialize(io, CausalFrames.JLSHEADER)
            serialize(io, DataFrame(time = [5, 6], x = [1, 2]))
            serialize(io, DataFrame(time = [3, 4], x = [3, 4]))
        end
        @test_throws ArgumentError load(ctx, readjls(unsorted))

        # a torn last record is reported as such, while the complete records
        # before it stay readable
        full = joinpath(dir, "full.jls")
        scan(ctx, p |> writejls(full))
        torn = joinpath(dir, "torn.jls")
        bytes = read(full)
        write(torn, bytes[1:(end-8)])
        @test_throws ArgumentError load(ctx, readjls(torn))
        @test DataFrame(load(ctx, readjls(torn) |> head(2))).time == [1, 3]
    end
end
