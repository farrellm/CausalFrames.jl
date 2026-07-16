using CausalFrames
using DataFrames
using Dates
using Tables
using Test

# A test-local multi-valued summarizer exercising the NamedTuple interface:
# tracks the minimum and maximum of a column.
mutable struct MinMax <: Summarizer
    column::Symbol
    lo::Any
    hi::Any
end
MinMax(column::Symbol) = MinMax(column, nothing, nothing)
CausalFrames.fresh(s::MinMax) = MinMax(s.column)
function CausalFrames.update!(s::MinMax, row)
    v = row[s.column]
    s.lo = s.lo === nothing ? v : min(s.lo, v)
    s.hi = s.hi === nothing ? v : max(s.hi, v)
    return nothing
end
CausalFrames.value(s::MinMax) =
    NamedTuple{(Symbol(s.column, :_min), Symbol(s.column, :_max))}((s.lo, s.hi))

# A test-local summarizer whose output column is illegally named :time.
mutable struct BadTime <: Summarizer end
CausalFrames.fresh(::BadTime) = BadTime()
CausalFrames.update!(::BadTime, row) = nothing
CausalFrames.value(::BadTime) = (; time = 0)

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
        twochunks = CausalPipeline() do ctx
            CausalFrame(ctx, [DataFrame(time = [1, 2, 2], qty = [1, 2, 3]),
                              DataFrame(time = [2, 3], qty = [4, 5])])
        end
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

        # chunk structure preserved, state carried across the boundary
        twochunks = CausalPipeline() do ctx
            CausalFrame(ctx, [DataFrame(time = [1, 2], qty = [1, 2]),
                              DataFrame(time = [3, 4], qty = [3, 4])])
        end
        frame = load(Context(0, 9), twochunks |> addsummarycolumns(Sum(:qty)))
        @test length(frame.chunks) == 2
        @test DataFrame(frame).qty_sum == [1, 3, 6, 10]

        # collision with an existing column
        @test_throws ArgumentError load(Context(0, 9),
            readcsv(path) |> addsummarycolumns(MinMax(:qty)) |>
                addsummarycolumns(MinMax(:qty)))
    end

    @testset "show" begin
        frame = load(Context(0, 6), clock(2))
        text = sprint(show, MIME("text/plain"), frame)
        @test occursin("CausalFrame{Int64} with 3 rows over [0, 6]", text)
    end
end
