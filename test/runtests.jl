using CausalFrames
using DataFrames
using Dates
using Tables
using Test

# A test-local multi-valued summarizer exercising the NamedTuple interface:
# tracks the minimum and maximum of a column. Also the worked example of the
# config/state split — the column rides in a type parameter, and the state's
# value type comes from the input schema.
struct MinMax{C} <: Summarizer end
MinMax(column::Symbol) = MinMax{column}()

mutable struct MinMaxState{C,L,H,T} <: SummarizerState
    seen::Bool
    lo::T
    hi::T
    MinMaxState{C,L,H,T}() where {C,L,H,T} = new{C,L,H,T}(false)
end

CausalFrames.emptyvalue(::MinMax{C}) where {C} =
    NamedTuple{(Symbol(C, :_min), Symbol(C, :_max))}((missing, missing))
CausalFrames.fresh(::MinMax{C}, intypes::NamedTuple) where {C} =
    MinMaxState{C,Symbol(C, :_min),Symbol(C, :_max),intypes[C]}()
CausalFrames.fresh(::MinMaxState{C,L,H,T}) where {C,L,H,T} = MinMaxState{C,L,H,T}()
function CausalFrames.update!(st::MinMaxState{C}, row) where {C}
    v = getproperty(row, C)
    st.lo = st.seen ? min(st.lo, v) : v
    st.hi = st.seen ? max(st.hi, v) : v
    st.seen = true
    return nothing
end
CausalFrames.value(st::MinMaxState{C,L,H,T}) where {C,L,H,T} =
    NamedTuple{(L, H),Tuple{T,T}}((st.lo, st.hi))

# A test-local summarizer whose output column is illegally named :time.
struct BadTime <: Summarizer end
CausalFrames.emptyvalue(::BadTime) = (; time = 0)

@testset "CausalFrames.jl" begin
    @testset "Context" begin
        ctx = Context(1, 10)
        @test ctx.start == 1
        @test ctx.stop == 10
        @test timetype(ctx) == Int
        @test timetype(Context(1, 10.0)) == Float64
        @test timetype(Context(DateTime(2026), DateTime(2027))) == DateTime
        @test_throws ArgumentError Context(10, 1)
    end

    @testset "emptyframe" begin
        frame = load(Context(1, 10), emptyframe())
        @test nrow(frame) == 0
        @test names(frame) == ["time"]
        df = DataFrame(frame)
        @test nrow(df) == 0
        @test eltype(df.time) == Int
    end

    @testset "clock" begin
        frame = load(Context(0, 10), clock(3))
        @test DataFrame(frame).time == [0, 3, 6, 9]

        # stop is excluded (half-open interval)
        frame = load(Context(0, 9), clock(3))
        @test DataFrame(frame).time == [0, 3, 6]

        frame = load(Context(DateTime(2026, 1, 1), DateTime(2026, 1, 1, 1)),
                     clock(Minute(15)))
        @test nrow(frame) == 4
        @test DataFrame(frame).time[end] == DateTime(2026, 1, 1, 0, 45)

        @test nrow(load(Context(5, 5), clock(1))) == 0
        @test_throws ArgumentError load(Context(0, 10), clock(0))
        @test_throws ArgumentError load(Context(0, 10), clock(-1))
    end

    @testset "readcsv" begin
        path = joinpath(mktempdir(), "ticks.csv")
        write(path, """
            time,bid,ask
            1,10.0,11.0
            3,10.5,11.5
            5,9.0,10.0
            7,9.5,10.5
            """)

        frame = load(Context(0, 100), readcsv(path))
        @test nrow(frame) == 4
        @test names(frame) == ["time", "bid", "ask"]

        # clipped to [start, stop)
        frame = load(Context(3, 7), readcsv(path))
        @test DataFrame(frame).time == [3, 5]

        # time column converted to the context's time type
        frame = load(Context(0.0, 100.0), readcsv(path))
        @test eltype(DataFrame(frame).time) == Float64

        unsorted = joinpath(mktempdir(), "unsorted.csv")
        write(unsorted, "time,x\n3,1\n1,2\n")
        @test_throws ArgumentError load(Context(0, 100), readcsv(unsorted))

        notime = joinpath(mktempdir(), "notime.csv")
        write(notime, "t,x\n1,2\n")
        @test_throws ArgumentError load(Context(0, 100), readcsv(notime))
    end

    @testset "frame invariants" begin
        ctx = Context(0, 10)
        @test_throws ArgumentError CausalFrame(ctx, DataFrame(x = [1, 2]))
        @test_throws ArgumentError CausalFrame(ctx, DataFrame(time = [3, 1]))
        @test_throws ArgumentError CausalFrame(ctx, DataFrame(time = [5, 11]))
        @test_throws ArgumentError CausalFrame(ctx, DataFrame(time = [-1, 5]))
        # stop itself is allowed (closed interval for frames)
        @test nrow(CausalFrame(ctx, DataFrame(time = [5, 10]))) == 2
        # non-decreasing across chunk boundaries
        @test_throws ArgumentError CausalFrame(ctx,
            [DataFrame(time = [4, 5]), DataFrame(time = [3])])
        # chunks must share a schema
        @test_throws ArgumentError CausalFrame(ctx,
            [DataFrame(time = [1], x = [1]), DataFrame(time = [2])])
    end

    @testset "filterrows and addcolumns" begin
        path = joinpath(mktempdir(), "ticks.csv")
        write(path, """
            time,bid,ask
            1,10.0,11.0
            3,10.5,11.5
            5,9.0,10.0
            7,9.5,10.5
            """)

        p = readcsv(path) |>
            filterrows(r -> r.bid >= 9.5) |>
            addcolumns(r -> (; mid = (r.bid + r.ask) / 2))
        df = DataFrame(load(Context(0, 100), p))
        @test df.time == [1, 3, 7]
        @test df.mid == [10.5, 11.0, 10.0]
        @test names(df) == ["time", "bid", "ask", "mid"]

        # row access by symbol and access to :time
        p = readcsv(path) |> filterrows(r -> r[:time] > 3)
        @test DataFrame(load(Context(0, 100), p)).time == [5, 7]

        # addcolumns may not touch time, and must return a NamedTuple
        bad = readcsv(path) |> addcolumns(r -> (; time = r.time))
        @test_throws ArgumentError load(Context(0, 100), bad)
        notuple = readcsv(path) |> addcolumns(r -> r.bid)
        @test_throws ArgumentError load(Context(0, 100), notuple)

        # transforms on an empty frame are no-ops
        p = emptyframe() |> filterrows(r -> true) |> addcolumns(r -> (; y = 1))
        @test nrow(load(Context(0, 100), p)) == 0
    end

    @testset "chunk concatenation property" begin
        p = clock(2) |>
            filterrows(r -> r.time % 4 == 0) |>
            addcolumns(r -> (; double = 2 * r.time))

        whole = load(Context(0, 20), p)
        left = load(Context(0, 10), p)
        right = load(Context(10, 20), p)
        stitched = CausalFrame(Context(0, 20),
            [DataFrame(left), DataFrame(right)])

        @test DataFrame(whole) == DataFrame(stitched)
        @test nrow(stitched) == nrow(left) + nrow(right)
        @test names(stitched) == ["time", "double"]
    end

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

    @testset "readcsv chunked" begin
        dir = mktempdir()
        path = joinpath(dir, "long.csv")
        open(path, "w") do io
            println(io, "time,x")
            for t in 1:50
                println(io, "$t,$(2t)")
            end
        end

        # small chunks stream as several frames but load identically
        p = readcsv(path; chunkbytes = 64)
        frames = collect(stream(Context(0, 100), p))
        @test length(frames) > 1
        @test DataFrame(load(Context(0, 100), p)) ==
              DataFrame(load(Context(0, 100), readcsv(path)))
        @test DataFrame(load(Context(0, 100), p)).time == 1:50

        # clipping to [start, stop) works across chunk boundaries
        @test DataFrame(load(Context(10, 20), p)).time == 10:19

        # early stop: disorder past the window is never read ...
        badtail = joinpath(dir, "badtail.csv")
        open(badtail, "w") do io
            println(io, "time,x")
            for t in 1:50
                println(io, "$t,$(2t)")
            end
            println(io, "7,0")   # unsorted, far past stop below
        end
        @test DataFrame(load(Context(0, 10),
            readcsv(badtail; chunkbytes = 64))).time == 1:9
        # ... but reading through it throws
        @test_throws ArgumentError load(Context(0, 1000),
            readcsv(badtail; chunkbytes = 64))

        @test_throws ArgumentError readcsv(path; chunkbytes = 0)
    end

    @testset "Tables.jl interface" begin
        frame = load(Context(0, 6), clock(2) |> addcolumns(r -> (; sq = r.time^2)))
        rows = collect(Tables.rows(frame))
        @test length(rows) == 3
        @test [r.sq for r in rows] == [0, 4, 16]
        @test Tables.columntable(frame).time == [0, 2, 4]
    end

    @testset "summarize" begin
        path = joinpath(mktempdir(), "trades.csv")
        write(path, """
            time,sym,qty
            1,a,10
            2,b,20
            2,a,30
            4,a,40
            """)

        # no key: one row at ctx.stop, input columns dropped
        frame = load(Context(0, 9), readcsv(path) |> summarize([Count(), Sum(:qty)]))
        df = DataFrame(frame)
        @test names(df) == ["time", "count", "qty_sum"]
        @test df.time == [9]
        @test df.count == [4]
        @test df.qty_sum == [100]

        # a single summarizer (not a collection) works too
        df = DataFrame(load(Context(0, 9), readcsv(path) |> summarize(Count())))
        @test df.count == [4]

        # multi-valued summarizer
        df = DataFrame(load(Context(0, 9), readcsv(path) |> summarize(MinMax(:qty))))
        @test names(df) == ["time", "qty_min", "qty_max"]
        @test df.qty_min == [10]
        @test df.qty_max == [40]

        # duplicate configurations dedup to one column
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize([Sum(:qty), Sum(:qty)])))
        @test names(df) == ["time", "qty_sum"]

        # min, max, first, last
        df = DataFrame(load(Context(0, 9), readcsv(path) |>
            summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)])))
        @test names(df) == ["time", "qty_min", "qty_max", "qty_first", "qty_last"]
        @test df.qty_min == [10]
        @test df.qty_max == [40]
        @test df.qty_first == [10]
        @test df.qty_last == [40]

        # sum of elements to the Nth power; :x_sum2 never dedups against :x_sum
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize([Sum(:qty), SumPower(:qty, 2)])))
        @test names(df) == ["time", "qty_sum", "qty_sum2"]
        @test df.qty_sum == [100]
        @test df.qty_sum2 == [3000]      # 10^2 + 20^2 + 30^2 + 40^2

        # power 1 is a distinct column from Sum, not a duplicate of it
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize([Sum(:qty), SumPower(:qty, 1)])))
        @test names(df) == ["time", "qty_sum", "qty_sum1"]
        @test df.qty_sum1 == [100]

        # no identity element: empty input summarizes to missing, not an error
        df = DataFrame(load(Context(0, 9), emptyframe() |>
            summarize([Min(:qty), Max(:qty), First(:qty), Last(:qty)])))
        @test df.time == [9]
        @test ismissing(only(df.qty_min))
        @test ismissing(only(df.qty_max))
        @test ismissing(only(df.qty_first))
        @test ismissing(only(df.qty_last))

        # keyed first/last pin down per-group ordering
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize([First(:qty), Last(:qty)]; key = :sym)))
        @test df.sym == ["a", "b"]
        @test df.qty_first == [10, 20]
        @test df.qty_last == [40, 20]

        # key: one row per unique key value, sorted by key
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize([Count(), Sum(:qty)]; key = :sym)))
        @test names(df) == ["time", "sym", "count", "qty_sum"]
        @test df.time == [9, 9]
        @test df.sym == ["a", "b"]
        @test df.count == [3, 1]
        @test df.qty_sum == [80, 20]

        # multi-column key
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarize(Count(); key = [:sym, :qty])))
        @test names(df) == ["time", "sym", "qty", "count"]
        @test df.sym == ["a", "a", "a", "b"]
        @test df.qty == [10, 30, 40, 20]
        @test all(df.count .== 1)

        # empty input: identity row without key, no rows with key
        df = DataFrame(load(Context(0, 9),
            emptyframe() |> summarize([Count(), Sum(:qty)])))
        @test df.time == [9]
        @test df.count == [0]
        @test df.qty_sum == [0]
        @test nrow(load(Context(0, 9),
            emptyframe() |> summarize(Count(); key = :sym))) == 0

        # validation
        @test_throws ArgumentError load(Context(0, 9),
            readcsv(path) |> summarize(BadTime()))
        @test_throws ArgumentError load(Context(0, 9),
            readcsv(path) |> summarize(Sum(:qty); key = :qty_sum))
        @test_throws ArgumentError load(Context(0, 9),
            readcsv(path) |> summarize(Summarizer[]))
    end

    @testset "summarizecycles" begin
        path = joinpath(mktempdir(), "trades.csv")
        write(path, """
            time,sym,qty
            1,a,10
            2,b,20
            2,a,30
            4,a,40
            """)

        # fresh state per cycle, one row per unique time
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarizecycles([Count(), Sum(:qty)])))
        @test names(df) == ["time", "count", "qty_sum"]
        @test df.time == [1, 2, 4]
        @test df.count == [1, 2, 1]
        @test df.qty_sum == [10, 50, 40]

        # per key within each cycle, sorted by key
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> summarizecycles(Sum(:qty); key = :sym)))
        @test df.time == [1, 2, 2, 4]
        @test df.sym == ["a", "a", "b", "a"]
        @test df.qty_sum == [10, 30, 20, 40]

        # a cycle spanning a chunk boundary is one cycle
        twochunks = CausalPipeline(ctx ->
            [DataFrame(time = [1, 2, 2], qty = [1, 2, 3]),
             DataFrame(time = [2, 3], qty = [4, 5])])
        df = DataFrame(load(Context(0, 9),
            twochunks |> summarizecycles([Count(), Sum(:qty)])))
        @test df.time == [1, 2, 3]
        @test df.count == [1, 3, 1]
        @test df.qty_sum == [1, 9, 5]

        # empty input yields no rows
        @test nrow(load(Context(0, 9), emptyframe() |> summarizecycles(Count()))) == 0
    end

    @testset "addsummarycolumns" begin
        path = joinpath(mktempdir(), "trades.csv")
        write(path, """
            time,sym,qty
            1,a,10
            2,b,20
            2,a,30
            4,a,40
            """)

        # running values after each row, existing columns kept
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> addsummarycolumns([Count(), Sum(:qty)])))
        @test names(df) == ["time", "sym", "qty", "count", "qty_sum"]
        @test df.count == [1, 2, 3, 4]
        @test df.qty_sum == [10, 30, 60, 100]

        # per-key running values
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> addsummarycolumns(Sum(:qty); key = :sym)))
        @test df.qty_sum == [10, 20, 40, 80]

        # a running min/last never sees an empty group, so never yields missing
        df = DataFrame(load(Context(0, 9),
            readcsv(path) |> addsummarycolumns([Min(:qty), Last(:qty)])))
        @test df.qty_min == [10, 10, 10, 10]
        @test df.qty_last == [10, 20, 30, 40]

        # chunk structure preserved, state carried across the boundary
        twochunks = CausalPipeline(ctx ->
            [DataFrame(time = [1, 2], qty = [1, 2]),
             DataFrame(time = [3, 4], qty = [3, 4])])
        p = twochunks |> addsummarycolumns(Sum(:qty))
        @test length(load(Context(0, 9), p).chunks) == 2
        frames = collect(stream(Context(0, 9), p))
        @test length(frames) == 2
        @test DataFrame(frames[1]).qty_sum == [1, 3]
        @test DataFrame(frames[2]).qty_sum == [6, 10]
        @test context(frames[1]) == Context(0, 3)
        @test context(frames[2]) == Context(3, 9)
        @test DataFrame(load(Context(0, 9), p)).qty_sum == [1, 3, 6, 10]

        # collision with an existing column
        @test_throws ArgumentError load(Context(0, 9),
            readcsv(path) |> addsummarycolumns(MinMax(:qty)) |>
                addsummarycolumns(MinMax(:qty)))
    end

    @testset "summarizer output types" begin
        # A summarizer's output column takes its element type from the input
        # column: Min/Max/First/Last verbatim, Sum/SumPower via Base.sum's own
        # widening rule.
        chunks(cs...) = CausalPipeline(ctx -> collect(cs))
        summarizeall(col) = DataFrame(load(Context(0, 9),
            chunks(DataFrame(time = [1, 2, 3], x = col)) |>
                summarize([Count(), Sum(:x), SumPower(:x, 2),
                           Min(:x), Max(:x), First(:x), Last(:x)])))

        # small ints widen the way Base.sum widens, extrema do not
        df = summarizeall(Int32[3, 1, 2])
        @test eltype(df.count) == Int
        @test eltype(df.x_sum) == Int64
        @test eltype(df.x_sum2) == Int64
        @test all(eltype(df[!, c]) == Int32
                  for c in [:x_min, :x_max, :x_first, :x_last])

        # the accumulator is widened up front, so it cannot overflow the way
        # summing in the input's own type would
        df = summarizeall(Int8[100, 100, 100])
        @test eltype(df.x_sum) == Int64
        @test only(df.x_sum) == 300
        @test eltype(df.x_min) == Int8

        # nothing to widen: Float32 sums as Float32
        df = summarizeall(Float32[3, 1, 2])
        @test all(eltype(df[!, c]) == Float32
                  for c in [:x_sum, :x_sum2, :x_min, :x_max, :x_first, :x_last])

        df = summarizeall(Bool[true, true, false])
        @test eltype(df.x_sum) == Int64 && only(df.x_sum) == 2
        @test eltype(df.x_min) == Bool

        # a missing-permitting column stays missing-permitting throughout,
        # whether or not the summarized value is itself missing
        df = summarizeall(Union{Missing,Int}[1, missing, 3])
        @test all(eltype(df[!, c]) == Union{Missing,Int64}
                  for c in [:x_sum, :x_min, :x_max, :x_first, :x_last])
        @test ismissing(only(df.x_min))     # min poisons
        @test only(df.x_first) == 1         # first does not

        # a source may infer a column's type per chunk, so the state widens
        # across the boundary rather than forcing the first chunk's type
        df = DataFrame(load(Context(0, 9),
            chunks(DataFrame(time = [1, 2], qty = Int[1, 2]),
                   DataFrame(time = [3, 4], qty = Float64[2.5, 3.5])) |>
                summarize([Sum(:qty), Min(:qty)])))
        @test eltype(df.qty_sum) == Float64 && only(df.qty_sum) == 9.0
        @test eltype(df.qty_min) == Float64 && only(df.qty_min) == 1.0

        # the same, with a key group table and an open cycle in flight over the
        # widening boundary
        mixed = chunks(DataFrame(time = [1, 1], sym = ["a", "b"], qty = Int[1, 2]),
                       DataFrame(time = [1, 2], sym = ["a", "b"],
                                 qty = Float64[2.5, 3.5]))
        df = DataFrame(load(Context(0, 9),
            mixed |> summarize([Sum(:qty), Min(:qty)]; key = :sym)))
        @test df.sym == ["a", "b"]
        @test eltype(df.qty_sum) == Float64 && df.qty_sum == [3.5, 5.5]
        @test eltype(df.qty_min) == Float64 && df.qty_min == [1.0, 2.0]

        df = DataFrame(load(Context(0, 9),
            mixed |> summarizecycles(Sum(:qty); key = :sym)))
        @test eltype(df.qty_sum) == Float64
        @test df.time == [1, 1, 2] && df.qty_sum == [3.5, 2.0, 3.5]

        df = DataFrame(load(Context(0, 9), mixed |> addsummarycolumns(Sum(:qty))))
        @test eltype(df.qty_sum) == Float64 && df.qty_sum == [1.0, 3.0, 5.5, 9.0]

        # the other two transforms type their columns the same way
        typed = chunks(DataFrame(time = [1, 1, 2], sym = ["a", "b", "a"],
                                 x = Int32[5, 7, 9]))
        for q in [summarize([Sum(:x), Min(:x)]; key = :sym),
                  summarizecycles([Sum(:x), Min(:x)]),
                  addsummarycolumns([Sum(:x), Min(:x)])]
            df = DataFrame(load(Context(0, 9), typed |> q))
            @test eltype(df.x_sum) == Int64
            @test eltype(df.x_min) == Int32
        end

        # per-frame types, which load's vcat would otherwise mask by promoting
        frames = collect(stream(Context(0, 9),
            chunks(DataFrame(time = [1], x = Int32[1]),
                   DataFrame(time = [2], x = Int32[2])) |>
                addsummarycolumns(Sum(:x))))
        @test all(eltype(DataFrame(f).x_sum) == Int64 for f in frames)

        # the property the typing rests on: a state's value is concretely
        # typed, so the column built from it is too
        st = CausalFrames.fresh(Sum(:x), (time = Int64, x = Int32))
        CausalFrames.update!(st, (time = 1, x = Int32(5)))
        @test @inferred(CausalFrames.value(st)) === (x_sum = Int64(5),)
        st = CausalFrames.fresh(Min(:x), (time = Int64, x = Int32))
        CausalFrames.update!(st, (time = 1, x = Int32(5)))
        @test @inferred(CausalFrames.value(st)) === (x_min = Int32(5),)
    end

    @testset "show" begin
        frame = load(Context(0, 6), clock(2))
        text = sprint(show, MIME("text/plain"), frame)
        @test occursin("CausalFrame{Int64} with 3 rows over [0, 6]", text)
    end
end
