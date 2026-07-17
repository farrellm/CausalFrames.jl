@testset "stream" begin
    # sub-contexts tile [start, stop): boundaries at the next chunk's
    # first time, last context stops at ctx.stop
    frames = collect(stream(Context(0, 10), clock(2; batchsize = 2)))
    @test length(frames) == 3
    @test [DataFrame(f).time for f in frames] == [[0, 2], [4, 6], [8]]
    @test [context(f) for f in frames] ==
          [Context(0, 4), Context(4, 8), Context(8, 10)]

    # laziness: nothing is pulled until iteration, then one chunk of
    # lookahead; the failing third chunk is only reached on demand
    pulled = Ref(0)
    lazyfail = CausalPipeline() do ctx
        (begin
             pulled[] += 1
             pulled[] <= 2 || throw(ArgumentError("chunk 3 pulled"))
             DataFrame(time = [i])
         end for i in 1:3)
    end
    it = stream(Context(0, 10), lazyfail)
    @test pulled[] == 0
    f1, st = iterate(it)
    @test pulled[] == 2
    @test DataFrame(f1).time == [1]
    @test_throws ArgumentError iterate(it, st)
    pulled[] = 0
    @test_throws ArgumentError load(Context(0, 10), lazyfail)

    # transforms are lazy too: composing and streaming pulls nothing
    pulled[] = 0
    p = lazyfail |> filterrows(r -> true) |> summarize(Count())
    it = stream(Context(0, 10), p)
    @test pulled[] == 0

    # cross-chunk time disorder is caught while streaming and by load
    disorder = CausalPipeline(ctx ->
        [DataFrame(time = [4, 5]), DataFrame(time = [3])])
    @test_throws ArgumentError collect(stream(Context(0, 9), disorder))
    @test_throws ArgumentError load(Context(0, 9), disorder)

    # load wraps the streamed chunks without copying; DataFrame(frame)
    # is the copy point
    chunk = DataFrame(time = [1, 2])
    frame = load(Context(0, 9), CausalPipeline(ctx -> [chunk]))
    @test only(frame.chunks) === chunk
    @test DataFrame(frame) !== chunk

    # empty streams yield no frames; load gives a zero-row time-only frame
    @test isempty(collect(stream(Context(0, 9), emptyframe())))
    @test isempty(collect(stream(Context(5, 5), clock(1))))
    allgone = clock(1) |> filterrows(r -> false)
    @test isempty(collect(stream(Context(0, 9), allgone)))
    @test nrow(load(Context(0, 9), allgone)) == 0
    @test names(load(Context(0, 9), allgone)) == ["time"]

    # chunks emptied by a filter are skipped; tiling stays sound
    sparse = clock(1; batchsize = 2) |> filterrows(r -> r.time in (0, 5))
    frames = collect(stream(Context(0, 8), sparse))
    @test [DataFrame(f).time for f in frames] == [[0], [5]]
    @test [context(f) for f in frames] == [Context(0, 5), Context(5, 8)]

    # summarize streams as a single frame at stop, over [start, stop]
    frames = collect(stream(Context(0, 9), clock(1) |> summarize(Count())))
    @test length(frames) == 1
    @test context(only(frames)) == Context(0, 9)
    @test DataFrame(only(frames)).time == [9]
    @test DataFrame(only(frames)).count == [9]
    # keyed summarize of an empty input streams no frames
    @test isempty(collect(stream(Context(0, 9),
        emptyframe() |> summarize(Count(); key = :sym))))

    # summarizecycles closes a cycle once a later time arrives; the open
    # cycle is buffered across the chunk boundary and flushed at the end
    twochunks = CausalPipeline(ctx ->
        [DataFrame(time = [1, 2, 2], qty = [1, 2, 3]),
         DataFrame(time = [2, 3], qty = [4, 5])])
    frames = collect(stream(Context(0, 9),
        twochunks |> summarizecycles([Count(), Sum(:qty)])))
    @test [DataFrame(f).time for f in frames] == [[1], [2], [3]]
    @test [only(DataFrame(f).count) for f in frames] == [1, 3, 1]
    @test [only(DataFrame(f).qty_sum) for f in frames] == [1, 9, 5]
end
