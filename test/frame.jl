@testset "emptyframe" begin
    frame = load(Context(1, 10), emptyframe())
    @test nrow(frame) == 0
    @test names(frame) == ["time"]
    df = DataFrame(frame)
    @test nrow(df) == 0
    @test eltype(df.time) == Int
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
    # rows are served through the column-access fallback
    rows = collect(Tables.rows(frame))
    @test length(rows) == 3
    @test [r.sq for r in rows] == [0, 4, 16]
    @test Tables.columntable(frame).time == [0, 2, 4]
    sch = Tables.schema(frame)
    @test sch.names == (:time, :sq)
    @test sch.types == (Int, Int)
end

@testset "Tables.schema promotes across chunks" begin
    ctx = Context(0, 10)
    frame = CausalFrame(ctx, [DataFrame(time = [1], x = [1]),
                              DataFrame(time = [2], x = [2.5])])
    sch = Tables.schema(frame)
    @test sch.names == (:time, :x)
    @test sch.types == (Int, Float64)
    @test eltype(DataFrame(frame).x) == Float64
end

@testset "empty frame Tables behavior" begin
    frame = load(Context(1, 10), emptyframe())
    sch = Tables.schema(frame)
    @test sch.names == (:time,)
    @test sch.types == (Int,)
    # partitions agree with DataFrame(frame): one zero-row time-only frame
    parts = collect(Tables.partitions(frame))
    @test length(parts) == 1
    @test parts[1] == DataFrame(frame)
end

@testset "show" begin
    frame = load(Context(0, 6), clock(2))
    text = sprint(show, MIME("text/plain"), frame)
    @test occursin("CausalFrame{Int64} with 3 rows over [0, 6]", text)
end
