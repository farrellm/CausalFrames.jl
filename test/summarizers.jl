@testset "summarizer output types" begin
    # A summarizer's output column takes its element type from the input
    # column: Min/Max/First/Last verbatim, Sum/SumPower via Base.sum's own
    # widening rule.
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    summarizeall(col) = DataFrame(load(Context(0, 9),
        chunks(DataFrame(time = [1, 2, 3], x = col)) |>
            summarize([Count(), Sum(:x), SumPower(:x, 2), Moment(:x, 2),
                       Min(:x), Max(:x), First(:x), Last(:x)])))

    # small ints widen the way Base.sum widens, extrema do not; a moment is
    # its power sum divided by the count
    df = summarizeall(Int32[3, 1, 2])
    @test eltype(df.count) == Int
    @test eltype(df.x_sum) == Int64
    @test eltype(df.x_sumpower_2) == Int64
    @test eltype(df.x_moment_2) == Float64
    @test all(eltype(df[!, c]) == Int32
              for c in [:x_min, :x_max, :x_first, :x_last])

    # the accumulator is widened up front, so it cannot overflow the way
    # summing in the input's own type would
    df = summarizeall(Int8[100, 100, 100])
    @test eltype(df.x_sum) == Int64
    @test only(df.x_sum) == 300
    @test eltype(df.x_min) == Int8

    # nothing to widen: Float32 sums as Float32, and divides to Float32
    df = summarizeall(Float32[3, 1, 2])
    @test all(eltype(df[!, c]) == Float32
              for c in [:x_sum, :x_sumpower_2, :x_moment_2,
                        :x_min, :x_max, :x_first, :x_last])

    df = summarizeall(Bool[true, true, false])
    @test eltype(df.x_sum) == Int64 && only(df.x_sum) == 2
    @test eltype(df.x_min) == Bool

    # a missing-permitting column stays missing-permitting throughout,
    # whether or not the summarized value is itself missing
    df = summarizeall(Union{Missing,Int}[1, missing, 3])
    @test all(eltype(df[!, c]) == Union{Missing,Int64}
              for c in [:x_sum, :x_min, :x_max, :x_first, :x_last])
    @test eltype(df.x_moment_2) == Union{Missing,Float64}
    @test ismissing(only(df.x_min))     # min poisons
    @test ismissing(only(df.x_moment_2))    # via its poisoned power sum
    @test only(df.x_first) == 1         # first does not

    # a source may infer a column's type per chunk, so the state widens
    # across the boundary rather than forcing the first chunk's type
    df = DataFrame(load(Context(0, 9),
        chunks(DataFrame(time = [1, 2], qty = Int[1, 2]),
               DataFrame(time = [3, 4], qty = Float64[2.5, 3.5])) |>
            summarize([Sum(:qty), Min(:qty), Moment(:qty, 1)])))
    @test eltype(df.qty_sum) == Float64 && only(df.qty_sum) == 9.0
    @test eltype(df.qty_min) == Float64 && only(df.qty_min) == 1.0
    @test eltype(df.qty_moment_1) == Float64 && only(df.qty_moment_1) == 2.25

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

    # the same property for a dependent summarizer: its dependencies' names
    # are baked into the state type, so the two-argument value infers, as
    # does the accumulate-then-project fold over an expanded prototype set
    st = CausalFrames.fresh(Moment(:x, 2), (time = Int64, x = Int32))
    @test @inferred(CausalFrames.value(st, (count = 2, x_sumpower_2 = Int64(8)))) ===
        (x_moment_2 = 4.0,)
    protos, requested = CausalFrames.prototypes(Summarizer[Moment(:x, 2)], Symbol[])
    states = map(s -> CausalFrames.fresh(s, (time = Int64, x = Int64)), protos)
    foreach(s -> CausalFrames.update!(s, (time = 1, x = 2)), states)
    @test @inferred(CausalFrames.summaryvalues(states, Val(requested))) ===
        (x_moment_2 = 4.0,)
end
