@testset "summarizer output types" begin
    # A summarizer's output column takes its element type from the input
    # column: Min/Max/First/Last verbatim, Sum/SumPower via Base.sum's own
    # widening rule.
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    summarizeall(col) = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = [1, 2, 3], x = col)) |>
            summarize([Count(), Sum(:x), SumPower(:x, 2), Moment(:x, 2),
                Min(:x), Max(:x), First(:x), Last(:x)])),
    )

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
    @test all(
        eltype(df[!, c]) == Float32
        for c in [:x_sum, :x_sumpower_2, :x_moment_2,
            :x_min, :x_max, :x_first, :x_last]
    )

    df = summarizeall(Bool[true, true, false])
    @test eltype(df.x_sum) == Int64 && only(df.x_sum) == 2
    @test eltype(df.x_min) == Bool

    # a missing-permitting column stays missing-permitting throughout,
    # whether or not the summarized value is itself missing
    df = summarizeall(Union{Missing,Int}[1, missing, 3])
    @test all(
        eltype(df[!, c]) == Union{Missing,Int64}
        for c in [:x_sum, :x_min, :x_max, :x_first, :x_last]
    )
    @test eltype(df.x_moment_2) == Union{Missing,Float64}
    @test ismissing(only(df.x_min))     # min poisons
    @test ismissing(only(df.x_moment_2))    # via its poisoned power sum
    @test only(df.x_first) == 1         # first does not

    # a column's type may differ per chunk, so the state widens across the
    # boundary rather than keeping the first chunk's type
    df = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = [1, 2], qty = Int[1, 2]),
                DataFrame(time = [3, 4], qty = Float64[2.5, 3.5])) |>
            summarize([Sum(:qty), Min(:qty), Moment(:qty, 1)])),
    )
    @test eltype(df.qty_sum) == Float64 && only(df.qty_sum) == 9.0
    @test eltype(df.qty_min) == Float64 && only(df.qty_min) == 1.0
    @test eltype(df.qty_moment_1) == Float64 && only(df.qty_moment_1) == 2.25

    # the same, with a key group table and an open cycle in flight over the
    # widening boundary
    mixed = chunks(DataFrame(time = [1, 1], sym = ["a", "b"], qty = Int[1, 2]),
        DataFrame(time = [1, 2], sym = ["a", "b"],
            qty = Float64[2.5, 3.5]))
    df = DataFrame(
        load(Context(0, 9),
            mixed |> summarize([Sum(:qty), Min(:qty)]; key = :sym)),
    )
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
    frames = collect(
        stream(Context(0, 9),
            chunks(DataFrame(time = [1], x = Int32[1]),
                DataFrame(time = [2], x = Int32[2])) |>
            addsummarycolumns(Sum(:x))),
    )
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

@testset "distinct counting" begin
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    summarized(p, ss; kwargs...) =
        DataFrame(load(Context(0, 9), p |> summarize(ss; kwargs...)))

    # the count is of values, not rows, and it is an Int whatever the input
    df = summarized(chunks(DataFrame(time = [1, 2, 3, 4], x = [1, 2, 2, 3])),
        [Count(), CountDistinct(:x)])
    @test only(df.count) == 4
    @test only(df.x_countdistinct) == 3
    @test eltype(df.x_countdistinct) == Int

    # missing is a value like any other, and does NOT poison the way the
    # accumulating summarizers do: a distinct count stays knowable, so the
    # output column is Int rather than Union{Missing, Int}
    df = summarized(
        chunks(
            DataFrame(time = [1, 2, 3, 4],
                x = Union{Missing,Int}[1, missing, 1, missing]),
        ),
        [CountDistinct(:x), Sum(:x)])
    @test only(df.x_countdistinct) == 2
    @test eltype(df.x_countdistinct) == Int
    @test ismissing(only(df.x_sum))            # the contrast: sum poisons
    @test eltype(df.x_sum) == Union{Missing,Int64}

    # a column's type may differ per chunk, so the state widens, and values
    # equal across the promotion count once (1 and 1.0 are one value)
    df = summarized(
        chunks(DataFrame(time = [1, 2], x = Int[1, 2]),
            DataFrame(time = [3, 4], x = Float64[1.0, 2.5])),
        CountDistinct(:x))
    @test only(df.x_countdistinct) == 3

    # strings, to pin that the state is typed from the schema and not numeric
    df = summarized(
        chunks(DataFrame(time = [1, 2, 3], s = ["a", "b", "a"])),
        CountDistinct(:s))
    @test only(df.s_countdistinct) == 2

    # no rows at all: the emptyvalue, reachable only through a keyless
    # summarize of an empty input
    df = summarized(emptyframe(), CountDistinct(:x))
    @test only(df.x_countdistinct) == 0

    # per key, and per cycle — the paths that zero and reuse a state tuple,
    # where a set that did not fully reset would leak into the next emission
    p = chunks(
        DataFrame(time = [1, 1, 1, 2, 2], k = ["a", "a", "b", "a", "a"],
            x = [1, 1, 7, 5, 6]),
    )
    df = summarized(p, CountDistinct(:x); key = :k)
    @test df.k == ["a", "b"]
    @test df.x_countdistinct == [3, 1]
    df = DataFrame(load(Context(0, 9), p |> summarizecycles(CountDistinct(:x))))
    @test df.x_countdistinct == [2, 2]

    # an empty interval emits the identity, not a stale count; the trailing
    # partial [4, 6) is the closelast row, timestamped at the context's stop
    df = DataFrame(
        load(Context(0, 6),
            chunks(DataFrame(time = [0, 0, 4], x = [1, 2, 9])) |>
            intervalize(clock(2), CountDistinct(:x); closelast = true)),
    )
    @test df.time == [2, 4, 6]
    @test df.x_countdistinct == [2, 0, 1]
end

@testset "statistical summarizers" begin
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    # x and y over three rows; the reference values are computed by hand:
    # sum(x)=6, prod(x)=6, sum(x^2)=14, mean(x)=2, and (corrected) var(x)=1;
    # dot(x,y)=19 and (corrected) cov(x,y)=-1.5.
    input(xcol, ycol) = chunks(DataFrame(time = [1, 2, 3], x = xcol, y = ycol))
    stats(xcol, ycol, ss) =
        DataFrame(load(Context(0, 9), input(xcol, ycol) |> summarize(ss)))

    # values, checked against the hand-computed references
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4],
        [Product(:x), DotProduct(:x, :y), Mean(:x), Variance(:x),
            Std(:x), Covariance(:x, :y)])
    @test only(df.x_product) == 6
    @test only(df.x_y_dotproduct) == 19
    @test only(df.x_mean) == 2.0
    @test only(df.x_variance) == 1.0
    @test only(df.x_std) == 1.0
    @test only(df.x_y_covariance) == -1.5

    # element types: Product/DotProduct widen like Sum; the statistical
    # dependents divide integers to Float64
    @test eltype(df.x_product) == Int64
    @test eltype(df.x_y_dotproduct) == Int64
    @test all(
        eltype(df[!, c]) == Float64
        for c in [:x_mean, :x_variance, :x_std, :x_y_covariance]
    )

    # Covariance(:x, :x) is Variance(:x)
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4], [Covariance(:x, :x)])
    @test only(df.x_x_covariance) == 1.0
    @test eltype(df.x_x_covariance) == Float64

    # Float32 stays Float32 throughout
    df = stats(Float32[3, 1, 2], Float32[2, 5, 4],
        [Product(:x), DotProduct(:x, :y), Mean(:x), Variance(:x),
            Std(:x), Covariance(:x, :y)])
    @test all(
        eltype(df[!, c]) == Float32
        for c in [:x_product, :x_y_dotproduct, :x_mean, :x_variance,
            :x_std, :x_y_covariance]
    )

    # corrected = false follows Statistics: divide by n rather than n - 1
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4],
        [Variance(:x; corrected = false), Std(:x; corrected = false),
            Covariance(:x, :y; corrected = false)])
    @test only(df.x_variance) ≈ 2 / 3
    @test only(df.x_std) ≈ sqrt(2 / 3)
    @test only(df.x_y_covariance) == -1.0

    # a single corrected sample is NaN (0/0), never a DivideError; uncorrected
    # is 0
    single(ss) = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = [1], x = Int[5])) |> summarize(ss)),
    )
    df = single([Variance(:x), Std(:x)])
    @test isnan(only(df.x_variance)) && isnan(only(df.x_std))
    df = single([Variance(:x; corrected = false)])
    @test only(df.x_variance) == 0.0

    # the accumulators widen up front, so per-row products cannot overflow the
    # way multiplying in the input's own type would
    df = stats(Int8[100, 100, 100], Int8[100, 100, 100],
        [Product(:x), DotProduct(:x, :y)])
    @test eltype(df.x_product) == Int64 && only(df.x_product) == 1_000_000
    @test eltype(df.x_y_dotproduct) == Int64 && only(df.x_y_dotproduct) == 30_000

    # missing-permitting input stays missing-permitting and poisons the value
    df = stats(Union{Missing,Int}[1, missing, 3], Int[2, 5, 4],
        [Product(:x), DotProduct(:x, :y), Mean(:x), Variance(:x),
            Std(:x), Covariance(:x, :y)])
    @test all(
        eltype(df[!, c]) == Union{Missing,Float64}
        for c in [:x_mean, :x_variance, :x_std, :x_y_covariance]
    )
    @test eltype(df.x_product) == Union{Missing,Int64}
    @test eltype(df.x_y_dotproduct) == Union{Missing,Int64}
    @test all(
        ismissing(only(df[!, c]))
        for c in [:x_product, :x_y_dotproduct, :x_mean, :x_variance,
            :x_std, :x_y_covariance]
    )

    # a column's type may widen across chunks, so the two-column accumulator
    # states widen too, carrying their accumulated total over
    df = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = [1, 2], x = Int[2, 3], y = Int[1, 2]),
                DataFrame(time = [3], x = Float64[4.0], y = Float64[0.5])) |>
            summarize([Product(:x), DotProduct(:x, :y)])),
    )
    @test eltype(df.x_product) == Float64 && only(df.x_product) == 24.0
    @test eltype(df.x_y_dotproduct) == Float64 && only(df.x_y_dotproduct) == 10.0

    # the value property the typing rests on: each dependent summarizer's
    # two-argument value is inferrable, with the divisor and dependency names
    # baked into the state type
    st = CausalFrames.fresh(Mean(:x), (time = Int64, x = Int32))
    @test @inferred(CausalFrames.value(st, (count = 3, x_sum = Int64(6)))) ===
          (x_mean = 2.0,)
    st = CausalFrames.fresh(Variance(:x), (time = Int64, x = Int32))
    @test @inferred(
        CausalFrames.value(st,
            (count = 3, x_sum = Int64(6), x_sumpower_2 = Int64(14)))
    ) ===
          (x_variance = 1.0,)
    st = CausalFrames.fresh(Std(:x), (time = Int64, x = Int32))
    @test @inferred(CausalFrames.value(st, (x_variance = 4.0,))) === (x_std = 2.0,)
    st = CausalFrames.fresh(Covariance(:x, :y), (time = Int64, x = Int32, y = Int32))
    @test @inferred(
        CausalFrames.value(st,
            (count = 3, x_sum = Int64(6), y_sum = Int64(11),
                x_y_dotproduct = Int64(19)))
    ) === (x_y_covariance = -1.5,)

    # nested dependency expansion (Std -> Variance -> raw sums) stays inferrable
    # through the accumulate-then-project fold, and hidden dependencies are
    # folded but not emitted
    protos, requested =
        CausalFrames.prototypes(Summarizer[Std(:x), Covariance(:x, :y)], Symbol[])
    @test requested === (:x_std, :x_y_covariance)
    states = map(s -> CausalFrames.fresh(s, (time = Int64, x = Int64, y = Int64)),
        protos)
    for (t, xv, yv) in [(1, 3, 2), (2, 1, 5)]
        foreach(s -> CausalFrames.update!(s, (time = t, x = xv, y = yv)), states)
    end
    @test @inferred(CausalFrames.summaryvalues(states, Val(requested))) ===
          (x_std = sqrt(2.0), x_y_covariance = -3.0)

    # Correlation: cov / (std * std), clamped to [-1, 1], following
    # Statistics.cor — no corrected keyword (the factor cancels)
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4], [Correlation(:x, :y)])
    @test only(df.x_y_correlation) ≈ -1.5 / (1.0 * sqrt(7 / 3))
    @test eltype(df.x_y_correlation) == Float64

    # a variable correlates perfectly with itself
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4], [Correlation(:x, :x)])
    @test only(df.x_x_correlation) ≈ 1.0

    # Float32 stays Float32
    df = stats(Float32[3, 1, 2], Float32[2, 5, 4], [Correlation(:x, :y)])
    @test eltype(df.x_y_correlation) == Float32

    # a single sample is NaN, never a DivideError or DomainError
    df = single([Correlation(:x, :x)])
    @test isnan(only(df.x_x_correlation))

    # missing-permitting input stays missing-permitting and poisons the value
    df = stats(Union{Missing,Int}[1, missing, 3], Int[2, 5, 4],
        [Correlation(:x, :y)])
    @test eltype(df.x_y_correlation) == Union{Missing,Float64}
    @test ismissing(only(df.x_y_correlation))

    # the dependent value infers, with the dependency names baked into the
    # state type
    st = CausalFrames.fresh(Correlation(:x, :y), (time = Int64, x = Int32, y = Int32))
    @test @inferred(
        CausalFrames.value(st,
            (x_y_covariance = -1.5, x_std = 1.0, y_std = 2.0))
    ) ===
          (x_y_correlation = -0.75,)
end

@testset "symmetric summarizers" begin
    # A summarizer symmetric in two columns folds under the isless-sorted
    # argument order but still emits the column the caller asked for, so
    # Σab and Σba cost one accumulator between them rather than two.
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    stats(xcol, ycol, ss) = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = [1, 2, 3], x = xcol, y = ycol)) |>
            summarize(ss)),
    )
    intypes = (time = Int64, x = Int32, y = Int32)
    # a prototype that folds per-row state, as opposed to a fieldless derived
    # one: what "shares the accumulator" means
    accumulators(protos) = [
        only(keys(CausalFrames.emptyvalue(s))) for s in protos
        if length(keys(CausalFrames.emptyvalue(s))) == 1 &&
            !(CausalFrames.fresh(s, intypes) isa CausalFrames.DerivedState)
    ]

    # the reversed request gets its own column, holding the same value
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4],
        [DotProduct(:x, :y), DotProduct(:y, :x),
            Covariance(:x, :y), Covariance(:y, :x)])
    @test only(df.y_x_dotproduct) == only(df.x_y_dotproduct) == 19
    @test only(df.y_x_covariance) == only(df.x_y_covariance) == -1.5
    @test eltype(df.y_x_dotproduct) == Int64
    @test eltype(df.y_x_covariance) == Float64

    # ... including on its own, with nothing canonical requested alongside
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4], [DotProduct(:y, :x)])
    @test only(df.y_x_dotproduct) == 19
    @test names(df) == ["time", "y_x_dotproduct"]
    df = stats(Float32[3, 1, 2], Float32[2, 5, 4],
        [DotProduct(:y, :x), Covariance(:y, :x)])
    @test eltype(df.y_x_dotproduct) == Float32
    @test eltype(df.y_x_covariance) == Float32

    # both orders fold one accumulator, the reversed one a fieldless rename
    # over it
    protos, requested =
        CausalFrames.prototypes(Summarizer[DotProduct(:x, :y),
            DotProduct(:y, :x)], Symbol[])
    @test requested === (:x_y_dotproduct, :y_x_dotproduct)
    @test accumulators(protos) == [:x_y_dotproduct]
    @test CausalFrames.fresh(DotProduct(:y, :x), intypes) isa CausalFrames.AliasState
    # a Covariance written either way reaches the same accumulator
    # (compared as a set: the two orders expand their Sums in their own order)
    for cov in (Covariance(:x, :y), Covariance(:y, :x))
        protos, _ = CausalFrames.prototypes(
            Summarizer[DotProduct(:x, :y), cov], Symbol[])
        @test Set(accumulators(protos)) ==
              Set([:x_y_dotproduct, :count, :x_sum, :y_sum])
        @test length(accumulators(protos)) == 4
    end
    # ... as does a Correlation, transitively through its Covariance
    protos, _ = CausalFrames.prototypes(
        Summarizer[DotProduct(:x, :y), Correlation(:y, :x)], Symbol[])
    @test count(==(:x_y_dotproduct), accumulators(protos)) == 1
    @test !(:y_x_dotproduct in accumulators(protos))

    # the alias renames its dependency's value and infers doing it
    st = CausalFrames.fresh(DotProduct(:y, :x), intypes)
    @test @inferred(CausalFrames.value(st, (x_y_dotproduct = Int64(19),))) ===
          (y_x_dotproduct = Int64(19),)
    protos, requested = CausalFrames.prototypes(
        Summarizer[DotProduct(:y, :x), Covariance(:y, :x)], Symbol[])
    states = map(s -> CausalFrames.fresh(s, (time = Int64, x = Int64, y = Int64)),
        protos)
    for (t, xv, yv) in [(1, 3, 2), (2, 1, 5), (3, 2, 4)]
        foreach(s -> CausalFrames.update!(s, (time = t, x = xv, y = yv)), states)
    end
    @test @inferred(CausalFrames.summaryvalues(states, Val(requested))) ===
          (y_x_dotproduct = 19, y_x_covariance = -1.5)

    # the value type comes from the dependency's declared field type, so a
    # missing-poisoned accumulator does not collapse the alias to Missing
    df = stats(Union{Missing,Int}[1, missing, 3], Int[2, 5, 4],
        [DotProduct(:y, :x), Covariance(:y, :x)])
    @test eltype(df.y_x_dotproduct) == Union{Missing,Int64}
    @test eltype(df.y_x_covariance) == Union{Missing,Float64}
    @test ismissing(only(df.y_x_dotproduct))

    # the reversed form keeps its declared structure, so a rolling window over
    # it stays on the running path (the differential lives in test/rolling.jl)
    @test CausalFrames.isinvertible(CausalFrames.fresh(DotProduct(:y, :x), intypes))
    @test CausalFrames.emptyvalue(DotProduct(:y, :x)) === (y_x_dotproduct = 0,)
    # fieldless, so zeroing preserves the type as for any other derived state
    st = CausalFrames.fresh(DotProduct(:y, :x), intypes)
    @test typeof(CausalFrames.fresh!(st)) === typeof(st)
end

@testset "linear regression" begin
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    # five points with a deliberate wobble, so the fit is not exact and the
    # standard error and t statistics have something to report
    xs = Float64[1, 2, 3, 4, 5]
    zs = Float64[1, 3, 2, 5, 4]
    ys = Float64[2.1, 3.9, 6.2, 7.8, 10.1]
    fit(ss; x = xs, z = zs, y = ys) = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = 1:length(x), x = x, z = z, y = y)) |>
            summarize(ss)),
    )
    rows(n, ss) = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = 1:n, x = xs[1:n], z = zs[1:n],
                y = ys[1:n])) |> summarize(ss)),
    )

    # simple regression against the textbook closed form, computed by hand here
    n = 5
    sx, sy = sum(xs), sum(ys)
    sxx = sum(xs .^ 2) - sx^2 / n
    sxy = sum(xs .* ys) - sx * sy / n
    syy = sum(ys .^ 2) - sy^2 / n
    beta = sxy / sxx
    alpha = sy / n - beta * sx / n
    sse = syy - beta * sxy
    sigma2 = sse / (n - 2)
    df = fit([LinearRegression(:x, :y)])
    @test only(df.x_beta) ≈ beta
    @test only(df.intercept_beta) ≈ alpha
    @test only(df.r2) ≈ 1 - sse / syy
    @test only(df.stderr) ≈ sqrt(sigma2)
    @test only(df.x_tstat) ≈ beta / sqrt(sigma2 / sxx)
    @test only(df.intercept_tstat) ≈
          alpha / sqrt(sigma2 * (1 / n + (sx / n)^2 / sxx))
    @test only(df.n) == 5

    # the output columns, exactly — this is the published contract, and
    # emptyvalue's keys are what the name-keyed deduplication works on
    outnames(s) = keys(CausalFrames.emptyvalue(s))
    @test outnames(LinearRegression([:x, :z], :y)) ===
          (:n, :r2, :stderr, :intercept_beta, :intercept_tstat,
        :x_beta, :x_tstat, :z_beta, :z_tstat)
    @test outnames(LinearRegression([:x, :z], :y; intercept = false)) ===
          (:n, :r2, :stderr, :x_beta, :x_tstat, :z_beta, :z_tstat)
    @test outnames(LinearRegression([:x, :z], :y; name = :m1)) ===
          (:m1_n, :m1_r2, :m1_stderr, :m1_intercept_beta, :m1_intercept_tstat,
        :m1_x_beta, :m1_x_tstat, :m1_z_beta, :m1_z_tstat)
    @test outnames(LinearRegression(:x, :y)) ===
          (:n, :r2, :stderr, :intercept_beta, :intercept_tstat,
        :x_beta, :x_tstat)
    # a string is one predictor name, not one name per character
    @test LinearRegression("price", :y) isa LinearRegression{(:price,),:y}
    # the predictor block follows the argument order, not sorted order
    @test outnames(LinearRegression([:z, :x], :y)) ===
          (:n, :r2, :stderr, :intercept_beta, :intercept_tstat,
        :z_beta, :z_tstat, :x_beta, :x_tstat)
    # ... and the emitted frame agrees with emptyvalue, in order
    df = fit([LinearRegression([:x, :z], :y; name = :m1)])
    @test Symbol.(Base.names(df)) ==
          [:time, collect(outnames(LinearRegression([:x, :z], :y; name = :m1)))...]

    # multiple regression: an exact fit y = 2x + 3z + 1 must be recovered
    ey = 2 .* xs .+ 3 .* zs .+ 1
    df = fit([LinearRegression([:x, :z], :y)]; y = ey)
    @test only(df.x_beta) ≈ 2
    @test only(df.z_beta) ≈ 3
    @test only(df.intercept_beta) ≈ 1
    @test only(df.r2) ≈ 1
    @test only(df.stderr) ≈ 0 atol = 1e-6

    # with orthogonal predictors each coefficient collapses to its own
    # univariate slope, which is the closed form checked above
    ox = Float64[1, 2, 3, 4]
    oz = Float64[1, -1, -1, 1]      # zero mean, and orthogonal to ox centered
    oy = Float64[1.0, 2.5, 4.0, 4.5]
    @test sum((ox .- sum(ox) / 4) .* (oz .- sum(oz) / 4)) == 0
    df = fit([LinearRegression([:x, :z], :y)]; x = ox, z = oz, y = oy)
    m = length(ox)
    sxxo = sum(ox .^ 2) - sum(ox)^2 / m
    szzo = sum(oz .^ 2) - sum(oz)^2 / m
    @test only(df.x_beta) ≈ (sum(ox .* oy) - sum(ox) * sum(oy) / m) / sxxo
    @test only(df.z_beta) ≈ (sum(oz .* oy) - sum(oz) * sum(oy) / m) / szzo

    # intercept = false: the uncentered fit, and no intercept columns
    df = fit([LinearRegression(:x, :y; intercept = false)])
    @test only(df.x_beta) ≈ sum(xs .* ys) / sum(xs .^ 2)
    @test !hasproperty(df, :intercept_beta)
    b0 = sum(xs .* ys) / sum(xs .^ 2)
    @test only(df.r2) ≈ 1 - (sum(ys .^ 2) - b0 * sum(xs .* ys)) / sum(ys .^ 2)

    # element types: the statistics divide integers to Float64 and keep
    # Float32, while n is Int either way
    df = fit([LinearRegression(:x, :y)]; x = Int32.(1:5), y = Int32[2, 4, 6, 8, 11])
    @test all(
        eltype(df[!, c]) == Float64
        for c in [:r2, :stderr, :intercept_beta, :intercept_tstat, :x_beta,
            :x_tstat]
    )
    @test eltype(df.n) == Int
    df = fit([LinearRegression(:x, :y)]; x = Float32.(xs), y = Float32.(ys))
    @test all(
        eltype(df[!, c]) == Float32
        for c in [:r2, :stderr, :intercept_beta, :intercept_tstat, :x_beta,
            :x_tstat]
    )
    @test eltype(df.n) == Int

    # degenerate windows report NaN, never a DivideError or DomainError — and
    # n is the honest row count throughout, so a poisoned fit still says how
    # much data it saw
    df = rows(1, [LinearRegression(:x, :y)])          # rank deficient
    @test only(df.n) == 1
    @test all(
        isnan(only(df[!, c]))
        for c in [:r2, :stderr, :intercept_beta, :intercept_tstat, :x_beta,
            :x_tstat]
    )
    df = rows(2, [LinearRegression(:x, :y)])          # exact fit, dof = 0
    @test only(df.n) == 2 && only(df.r2) ≈ 1
    @test isnan(only(df.stderr)) && isnan(only(df.x_tstat))
    @test isfinite(only(df.x_beta)) && isfinite(only(df.intercept_beta))
    df = fit([LinearRegression(:x, :y)]; y = fill(4.0, 5))   # constant response
    @test isnan(only(df.r2))
    df = fit([LinearRegression(:x, :y)]; x = fill(1.0, 5))   # constant predictor
    @test isnan(only(df.x_beta)) && isnan(only(df.r2))
    df = fit([LinearRegression([:x, :z], :y)]; z = 2 .* xs)  # collinear
    @test only(df.n) == 5
    @test all(
        isnan(only(df[!, c]))
        for c in [:r2, :stderr, :intercept_beta, :x_beta, :z_beta]
    )

    # missing-permitting input stays missing-permitting and poisons every
    # statistic, but never n
    df = fit([LinearRegression(:x, :y)];
        x = Union{Missing,Float64}[1, 2, missing, 4, 5])
    @test all(
        eltype(df[!, c]) == Union{Missing,Float64}
        for c in [:r2, :stderr, :intercept_beta, :intercept_tstat, :x_beta,
            :x_tstat]
    )
    @test all(
        ismissing(only(df[!, c]))
        for c in [:r2, :stderr, :intercept_beta, :intercept_tstat, :x_beta,
            :x_tstat]
    )
    @test eltype(df.n) == Int && only(df.n) == 5

    # no rows: missing statistics, but a count of zero rather than missing
    df = DataFrame(load(Context(0, 9), chunks() |> summarize(
        [LinearRegression(:x, :y)])))
    @test only(df.n) == 0 && ismissing(only(df.r2))

    # the two-argument value infers, on both solve paths and with the
    # intercept flag baked into the state type
    intypes = (time = Int64, x = Int32, z = Int32, y = Int32)
    st = CausalFrames.fresh(LinearRegression(:x, :y), intypes)
    @test @inferred(
        CausalFrames.value(st,
            (count = 5, x_sumpower_2 = Int64(55), y_sumpower_2 = Int64(200),
                x_y_dotproduct = Int64(100), x_sum = Int64(15),
                y_sum = Int64(30)))
    ).x_beta ≈ 1.0
    st = CausalFrames.fresh(LinearRegression(:x, :y; intercept = false), intypes)
    @test @inferred(
        CausalFrames.value(st,
            (count = 5, x_sumpower_2 = Int64(55), y_sumpower_2 = Int64(200),
                x_y_dotproduct = Int64(100)))
    ).x_beta ≈ 100 / 55
    st = CausalFrames.fresh(LinearRegression([:x, :z], :y), intypes)
    @test @inferred(
        CausalFrames.value(st,
            (count = 5, x_sumpower_2 = Int64(55), z_sumpower_2 = Int64(55),
                y_sumpower_2 = Int64(200), x_z_dotproduct = Int64(50),
                x_y_dotproduct = Int64(100), y_z_dotproduct = Int64(90),
                x_sum = Int64(15), z_sum = Int64(15), y_sum = Int64(30)))
    ) isa
          NamedTuple

    # two regressions over the same predictors share one accumulator per cross
    # product
    protos, requested = CausalFrames.prototypes(
        Summarizer[LinearRegression([:x, :z], :y; name = :m1),
            LinearRegression([:x, :z], :w; name = :m2)], Symbol[])
    folded = [
        only(keys(CausalFrames.emptyvalue(s))) for s in protos
        if length(keys(CausalFrames.emptyvalue(s))) == 1
    ]
    @test count(==(:count), folded) == 1
    @test count(==(:x_z_dotproduct), folded) == 1
    @test count(==(:x_sumpower_2), folded) == 1
    @test count(==(:x_sum), folded) == 1
    # the response-specific work is not shared, and appears once each
    @test count(==(:x_y_dotproduct), folded) == 1
    @test count(==(:w_x_dotproduct), folded) == 1
    @test requested === (outnames(LinearRegression([:x, :z], :y; name = :m1))...,
        outnames(LinearRegression([:x, :z], :w; name = :m2))...)

    # a regression shares with the statistical summarizers too: the squared
    # term goes through SumPower, the cross product through the canonically
    # (sorted) ordered DotProduct
    protos, _ = CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y; name = :m1), Variance(:y),
            Covariance(:x, :y)], Symbol[])
    folded = [
        only(keys(CausalFrames.emptyvalue(s))) for s in protos
        if length(keys(CausalFrames.emptyvalue(s))) == 1
    ]
    @test count(==(:y_sumpower_2), folded) == 1
    @test count(==(:x_y_dotproduct), folded) == 1
    @test count(==(:x_sum), folded) == 1
    # ... and the sharing survives a Covariance written the other way round,
    # since every symmetric summarizer folds under the canonical order
    protos, _ = CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y; name = :m1), Covariance(:y, :x)],
        Symbol[])
    folded = [
        only(keys(CausalFrames.emptyvalue(s))) for s in protos
        if length(keys(CausalFrames.emptyvalue(s))) == 1
    ]
    @test count(==(:x_y_dotproduct), folded) == 1
    @test !(:y_x_dotproduct in folded)

    # the whole accumulate-then-project fold stays inferrable
    protos, requested = CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y)], Symbol[])
    states = map(s -> CausalFrames.fresh(s, (time = Int64, x = Int64, y = Int64)),
        protos)
    for (t, xv, yv) in [(1, 1, 3), (2, 2, 5), (3, 3, 7)]
        foreach(s -> CausalFrames.update!(s, (time = t, x = xv, y = yv)), states)
    end
    out = @inferred CausalFrames.summaryvalues(states, Val(requested))
    @test out.x_beta ≈ 2.0 && out.intercept_beta ≈ 1.0 && out.n == 3

    # constructor and collision errors
    @test_throws ArgumentError LinearRegression(Symbol[], :y)
    @test_throws ArgumentError LinearRegression([:x, :x], :y)
    # two un-prefixed regressions collide on n/r2/stderr even with disjoint
    # predictors — which is exactly why `name` exists
    @test_throws ArgumentError CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y), LinearRegression(:z, :y)],
        Symbol[])
    # ... as does one regression against itself with the intercept flipped
    @test_throws ArgumentError CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y; name = :m1),
            LinearRegression(:x, :y; name = :m1, intercept = false)], Symbol[])
    # ... while a predictor named after a model-level column is rejected at
    # construction, rather than surfacing as a NamedTuple field-name error
    @test_throws ArgumentError LinearRegression([:x, :intercept], :y)
    # ... and only that one: the other model-level names take no suffix, so a
    # predictor sharing one of them produces distinct columns
    @test outnames(LinearRegression([:x, :r2], :y)) ===
          (:n, :r2, :stderr, :intercept_beta, :intercept_tstat,
        :x_beta, :x_tstat, :r2_beta, :r2_tstat)
    # an `intercept` predictor is fine once there is no constant term to clash
    # with, or once a prefix separates them
    @test outnames(LinearRegression([:x, :intercept], :y; intercept = false)) ===
          (:n, :r2, :stderr, :x_beta, :x_tstat,
        :intercept_beta, :intercept_tstat)
end

@testset "monoid and group structure" begin
    intypes = (time = Int64, x = Int64, y = Int64)
    rows = [(time = 1, x = 3, y = 2), (time = 2, x = 1, y = 7),
        (time = 3, x = 4, y = 1), (time = 4, x = 1, y = 8)]
    function fold(s, rs)
        st = CausalFrames.fresh(s, intypes)
        foreach(r -> CausalFrames.update!(st, r), rs)
        return st
    end

    # the hierarchy: the accumulators, the dependent summarizers, and — through
    # their windowed states — the trackers and CountDistinct are groups;
    # Product is a monoid only, as is the MinMax fixture; a plain Summarizer
    # is neither
    @test all(s -> s isa GroupSummarizer,
        [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
            AgeWeightedSum(:x), Moment(:x, 2), Mean(:x), Variance(:x), Std(:x),
            Covariance(:x, :y), Correlation(:x, :y),
            LinearRegression(:x, :y), LinearRegression([:x, :y], :y),
            Min(:x), Max(:x), First(:x), Last(:x), CountDistinct(:x)])
    @test all(s -> s isa MonoidSummarizer && !(s isa GroupSummarizer),
        [Product(:x), MinMax(:x)])
    @test !(Opaque(Sum(:x)) isa MonoidSummarizer)

    monoids = [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        AgeWeightedSum(:x), Product(:x), Min(:x), Max(:x), First(:x), Last(:x),
        MinMax(:x), CountDistinct(:x)]

    # fresh! must be indistinguishable from fresh: the transforms reuse state
    # tuples per cycle, interval and window query, so an incomplete reset leaks
    # values into the next emission. MinMax doesn't implement fresh!, which
    # exercises the `fresh(st)` default.
    selfcontained = [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        AgeWeightedSum(:x), Product(:x), Min(:x), Max(:x), First(:x), Last(:x),
        MinMax(:x), CountDistinct(:x), Opaque(Sum(:x))]
    for s in selfcontained
        reused = CausalFrames.fresh!(fold(s, rows))   # folded, then zeroed
        rebuilt = CausalFrames.fresh(s, intypes)      # never folded
        @test typeof(reused) === typeof(rebuilt)
        @test CausalFrames.isinvertible(reused) ==
              CausalFrames.isinvertible(rebuilt)
        # folding the same rows into each must now give the same summary
        foreach(r -> CausalFrames.update!(reused, r), rows)
        foreach(r -> CausalFrames.update!(rebuilt, r), rows)
        @test isequal(CausalFrames.value(reused), CausalFrames.value(rebuilt))
        # ... and so must a *different*, shorter run: a value leaked from the
        # first fold only shows up against a summary its rows could dominate
        reused = CausalFrames.fresh!(reused)
        rebuilt = CausalFrames.fresh(s, intypes)
        CausalFrames.update!(reused, rows[2])
        CausalFrames.update!(rebuilt, rows[2])
        @test isequal(CausalFrames.value(reused), CausalFrames.value(rebuilt))
    end

    # the dependent summarizers carry no state of their own, so fresh! only
    # has to preserve the type (their value comes from `vals` at emission)
    for s in [Moment(:x, 2), Mean(:x), Variance(:x), Std(:x),
        Covariance(:x, :y), Correlation(:x, :y), LinearRegression(:x, :y),
        TestVar(:x)]
        st = CausalFrames.fresh(s, intypes)
        @test typeof(CausalFrames.fresh!(st)) === typeof(st)
    end

    # Zeroing a built-in state allocates nothing, which every reuse path relies
    # on.
    for s in [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        AgeWeightedSum(:x), Product(:x), Min(:x), Max(:x), First(:x), Last(:x),
        CountDistinct(:x)]
        st = fold(s, rows)
        CausalFrames.fresh!(st)
        # CountDistinct is in this list because `empty!` keeps a Set's slots:
        # zeroing it is a write per slot, not a new allocation.
        @test (@allocated CausalFrames.fresh!(st)) == 0
    end

    # And over a whole tuple the cost doesn't grow with the number of resets.
    # Asserted as a slope rather than `@allocated(one call) == 0`: a lone
    # `@allocated` puts a function boundary around the call, where Julia 1.10
    # materializes the returned tuple that it elides once `freshall!` is
    # inlined into a folding loop, as the package always calls it.
    function resettotal(states, row, n)
        acc = 0
        for _ in 1:n
            states = CausalFrames.freshall!(states)
            CausalFrames.updateall!(states, row)
            acc += states[1].n
        end
        return acc
    end
    let states = map(s -> CausalFrames.fresh(s, intypes), (Count(), Sum(:x)))
        resettotal(states, first(rows), 2)
        base = @allocated resettotal(states, first(rows), 100)
        @test (@allocated resettotal(states, first(rows), 1000)) <= base + 100
    end

    # combine!(dest, a, b) equals folding a's rows then b's rows, for every
    # split — including an empty (fresh, identity) side — and tolerates dest
    # aliasing either argument
    for s in monoids
        whole = CausalFrames.value(fold(s, rows))
        for k in 0:length(rows)
            a = fold(s, rows[1:k])
            b = fold(s, rows[(k+1):end])
            dest = CausalFrames.fresh(a)
            @test @inferred(CausalFrames.combine!(dest, a, b)) === nothing
            @test CausalFrames.value(dest) == whole
        end
        a = fold(s, rows[1:2])
        CausalFrames.combine!(a, a, fold(s, rows[3:4]))
        @test CausalFrames.value(a) == whole
        b = fold(s, rows[3:4])
        CausalFrames.combine!(b, fold(s, rows[1:2]), b)
        @test CausalFrames.value(b) == whole
    end

    # associativity, on uneven splits: (r1 ⊕ r23) ⊕ r4 == r1 ⊕ (r23 ⊕ r4)
    for s in monoids
        p1, p2, p3 = fold(s, rows[1:1]), fold(s, rows[2:3]), fold(s, rows[4:4])
        l = CausalFrames.fresh(p1)
        CausalFrames.combine!(l, p1, p2)
        CausalFrames.combine!(l, l, p3)
        r = CausalFrames.fresh(p1)
        CausalFrames.combine!(r, p2, p3)
        CausalFrames.combine!(r, p1, r)
        @test CausalFrames.value(l) == CausalFrames.value(r)
    end

    # combining two identities stays an identity (the unseen tracker case),
    # and the state remains usable afterwards
    st = CausalFrames.fresh(Min(:x), intypes)
    CausalFrames.combine!(st, CausalFrames.fresh(st), CausalFrames.fresh(st))
    @test !st.seen
    CausalFrames.update!(st, rows[1])
    @test CausalFrames.value(st) == (x_min = 3,)

    # the group inverse: downdating the oldest rows equals folding the rest —
    # exact for integer accumulators
    # (AgeWeightedSum's is walked far harder under "age-weighted sum")
    for s in [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y)]
        st = fold(s, rows)
        @test @inferred(CausalFrames.downdate!(st, rows[1])) === nothing
        CausalFrames.downdate!(st, rows[2])
        @test CausalFrames.value(st) == CausalFrames.value(fold(s, rows[3:4]))
    end

    # a float accumulator inverts approximately (compensation shrinks but
    # does not eliminate the round-trip error in general) ...
    fst = CausalFrames.fresh(Sum(:x), (time = Int64, x = Float64))
    CausalFrames.update!(fst, (time = 1, x = 0.1))
    CausalFrames.update!(fst, (time = 2, x = 0.2))
    CausalFrames.downdate!(fst, (time = 2, x = 0.2))
    @test CausalFrames.value(fst).x_sum ≈ 0.1

    # ... and exactly when every intermediate value is representable
    dst = CausalFrames.fresh(Sum(:x), (time = Int64, x = Float64))
    for v in (0.5, 0.25, 0.125)
        CausalFrames.update!(dst, (; x = v))
    end
    CausalFrames.downdate!(dst, (; x = 0.5))
    CausalFrames.downdate!(dst, (; x = 0.25))
    @test CausalFrames.value(dst) === (x_sum = 0.125,)

    # the derived states are fieldless, so both operations are no-ops
    for s in [Mean(:x), Variance(:x), Correlation(:x, :y)]
        st = CausalFrames.fresh(s, intypes)
        @test CausalFrames.combine!(st, st, st) === nothing
        @test CausalFrames.downdate!(st, rows[1]) === nothing
    end

    # invertibility is a property of the realized accumulator type: the sum
    # family counts missing terms rather than folding them in (see below), so
    # even a missing-permitting accumulator stays invertible
    @test CausalFrames.isinvertible(CausalFrames.fresh(Sum(:x), intypes))
    @test CausalFrames.isinvertible(CausalFrames.fresh(Count(), intypes))
    mintypes = (time = Int64, x = Union{Missing,Int}, y = Union{Missing,Int})
    for s in [Sum(:x), SumPower(:x, 2), DotProduct(:x, :y)]
        @test CausalFrames.isinvertible(CausalFrames.fresh(s, mintypes))
    end
end

@testset "SumPower term specialization" begin
    # SumPower(c, 1) and SumPower(c, 2) fold terms whose exponent is in the type
    # (a move and a multiply) rather than PowerTerm's runtime `^`. That must not
    # change the output name, the accumulator type or the term's *bits*. See
    # notes/sumpower-terms.md.
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    intypes = (time = Int64, x = Float64)

    # the specialized exponents pick the cheap terms, everything else does not
    termof(n) = CausalFrames.fresh(SumPower(:x, n), intypes).term
    @test termof(1) isa CausalFrames.ColumnTerm{:x}
    @test termof(2) isa CausalFrames.PairProductTerm{:x,:x}
    @test termof(0) isa CausalFrames.PowerTerm{:x}
    @test termof(3) isa CausalFrames.PowerTerm{:x}
    # ... and the output column is untouched: SumPower(:x, 1) is still its own
    # column, distinct from Sum(:x), and SumPower(:x, 2) from DotProduct(:x, :x)
    @test keys(CausalFrames.emptyvalue(SumPower(:x, 1))) === (:x_sumpower_1,)
    @test keys(CausalFrames.emptyvalue(SumPower(:x, 2))) === (:x_sumpower_2,)

    # the accumulator type must not move, or the output column's eltype does.
    # This is the identity the specialization rests on, over every type the
    # package admits — including the Missing unions and the widening integers.
    for T in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt64, Bool,
        Float16, Float32, Float64, BigInt, BigFloat, Rational{Int},
        Union{Missing,Int64}, Union{Missing,Float64}, Union{Missing,Int8})
        @test CausalFrames.powertype(T, 1) === CausalFrames.sumtype(T)
        @test CausalFrames.powertype(T, 2) === CausalFrames.dottype(T, T)
    end

    # The exponent MUST stay in a variable here. Julia's parser rewrites a
    # *literal* exponent to Base.literal_pow(^, x, Val(2)), which for Float64
    # is `x * x` — so `@test same(x^2, x * x)` compares x*x with itself and
    # proves nothing. Only `runtimepow` reaches ^(::Float64, ::Int), the
    # algorithm these terms actually replace. Do not "simplify" it back.
    runtimepow(x, n::Int) = x^n
    bits(x) = reinterpret(Unsigned, x)
    same(a, b) = (isnan(a) && isnan(b)) || bits(a) === bits(b)
    edge = Float64[0.0, -0.0, 1.0, -1.0, 0.5, Inf, -Inf, NaN, 5.0e-324,
        floatmin(Float64), -floatmin(Float64), floatmax(Float64), 1e308, 1e-308,
        nextfloat(0.0), prevfloat(0.0), 3.141592653589793, -2.718281828459045]
    # a deterministic spread of bit patterns, without pulling Random into the
    # test dependencies: an LCG's high bits are good, and it is the exponent
    # field that has to vary here
    function bitpatterns(n)
        out = Vector{Float64}(undef, n)
        s = 0x2545f4914f6cdd1d
        for i in 1:n
            s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
            out[i] = reinterpret(Float64, s)
        end
        return out
    end
    rnd = bitpatterns(50_000)

    # n = 1 is exactly the runtime power, except that on Julia 1.10
    # `^(::Float64, ::Integer)` drops the sign of zero ((-0.0)^1 is 0.0; fixed
    # in 1.11). ColumnTerm returns the value untouched, so it is the more
    # correct there, and the difference is unobservable in output, since the
    # accumulator starts at +0.0 (asserted end to end below). Float32 is
    # unaffected.
    dropssignedzero(x) = VERSION < v"1.11" && x === -0.0
    @test all(x -> dropssignedzero(x) || same(runtimepow(x, 1), x), edge)
    @test all(x -> same(runtimepow(x, 1), x), rnd)
    # pin the quirk itself, so a change in either direction is noticed
    @test VERSION < v"1.11" ? runtimepow(-0.0, 1) === 0.0 :
          same(runtimepow(-0.0, 1), -0.0)
    @test all(x -> same(runtimepow(x, 1), x),
        Float32[0.0f0, -0.0f0, Inf32, -Inf32, NaN32, floatmin(Float32),
            floatmax(Float32), 1.0f-45])
    ints = Int64[0, 1, -1, 2, -2, 127, -128, 3037000499, -3037000499,
        typemax(Int64), typemin(Int64)]
    @test all(x -> runtimepow(x, 1) === x, ints)
    @test all(x -> runtimepow(x, 2) === x * x, ints)   # incl. the wrap-around
    @test all(x -> runtimepow(x, 2) === x * x, (true, false))

    # n = 2 is *not* bit-identical to the runtime power over floats: x * x is
    # the correctly rounded square. Assert that, and bound the disagreement to
    # one ULP; a real bug (a wrong exponent, a dropped convert) would miss by
    # far more. Where the runtime ^ misses is Base's business and not asserted:
    # near underflow its last bit depends on the CPU's FMA support.
    setprecision(BigFloat, 512) do
        @test all(x -> same(x * x, Float64(BigFloat(x)^2)), rnd)
        # concrete inputs whose square lands just above floatmin, where the two
        # can disagree — pinned rather than searched for
        nearunderflow = Float64[-2.6128464698398773e-154, 4.055521534318182e-154,
            2.3402388344754422e-154, -3.731470536494672e-154, 4.20214231599087e-154]
        @test all(x -> same(x * x, Float64(BigFloat(x)^2)), nearunderflow)
        @test all(x -> abs(runtimepow(x, 2) - x * x) <= eps(x * x), nearunderflow)
        @test all(x -> floatmin(Float64) < abs(x * x) < 1e-300, nearunderflow)
    end
    # wherever the two disagree it is by at most one ULP, near underflow
    differing = [x for x in rnd if !same(runtimepow(x, 2), x * x)]
    @test all(x -> abs(runtimepow(x, 2) - x * x) <= eps(x * x), differing)
    @test all(x -> abs(x * x) < 1e-300, differing)

    # The compensated accumulators classify NaN and ±Inf *terms* and carry the
    # sign of zero, so those bits must not move. None do, apart from Julia
    # 1.10's (-0.0)^1 above, which the +0.0 starting total absorbs.
    @test all(x -> same(runtimepow(x, 2), x * x), edge)
    @test only(
        DataFrame(
            load(Context(0, 9),
                chunks(DataFrame(time = [1, 2], x = [-0.0, -0.0])) |>
                summarize([SumPower(:x, 1)])),
        ).x_sumpower_1,
    ) === 0.0

    # End to end, the specialized fold must agree with the general PowerTerm
    # one over the values the compensated classifier cares about. The reference
    # is that same accumulator folding PowerTerm — not `Base.sum`, which sums
    # naively where these states compensate ([0.1, 0.2, 0.3] is exactly 0.6
    # here and 0.6000000000000001 there). `isequal` is the right comparison:
    # it separates -0.0 from 0.0 and matches NaN to NaN.
    #
    # The equality holds because every column below is well scaled. It is not a
    # universal property: at n = 2 the two folds differ by one ULP per term
    # once a square lands near underflow (see above), so a column of ~1e-160
    # would legitimately fail this.
    summed(col, n) = only(
        DataFrame(
            load(Context(0, 9),
                chunks(DataFrame(time = 1:length(col), x = col)) |>
                summarize([SumPower(:x, n)])),
        )[
            !,
            Symbol(:x_sumpower_, n),
        ],
    )
    function general(col, n)
        raw = CausalFrames.accumfresh(CausalFrames.PowerTerm{:x}(n),
            Symbol(:x_sumpower_, n), CausalFrames.powertype(eltype(col), n))
        foreach(x -> CausalFrames.update!(raw, (; x)), col)
        return only(CausalFrames.value(raw))
    end
    for col in (Float64[1.5, -0.0, 2.5], Float64[-0.0, -0.0], Float64[1.0, Inf, 2.0],
        Float64[1.0, -Inf, Inf], Float64[1.0, NaN, 2.0],
        Float64[1e300, 1e300, -1e300], Float64[0.1, 0.2, 0.3],
        Union{Missing,Float64}[1.5, missing, 2.5], Int64[3, -4, 5],
        Int32[3, -4, 5], Int8[100, 100, 100], Float32[1.5, -2.5, 3.5])
        for n in (1, 2)
            @test isequal(summed(col, n), general(col, n))
        end
    end
end

@testset "compensated float summation" begin
    chunks(cs...) = CausalPipeline(ctx -> collect(cs))
    ftypes = (time = Int64, x = Float64, y = Float64)
    function fold(s, xs)
        st = CausalFrames.fresh(s, ftypes)
        foreach(x -> CausalFrames.update!(st, (; x)), xs)
        return st
    end

    # float accumulators get the compensated state; a missing-permitting float
    # gets the compensated *counting* state over the non-missing type; BigFloat
    # keeps the plain one
    xterm = CausalFrames.ColumnTerm{:x}
    Comp64 = CausalFrames.Compensated{Float64}
    @test CausalFrames.fresh(Sum(:x), ftypes) isa
          CausalFrames.AccumState{:x_sum,Float64,xterm,false,Comp64}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = Union{Missing,Float64})) isa
          CausalFrames.AccumState{:x_sum,Float64,xterm,true,Comp64}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = Union{Missing,Int})) isa
          CausalFrames.AccumState{:x_sum,Int,xterm,true,Int}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = BigFloat)) isa
          CausalFrames.AccumState{:x_sum,BigFloat,xterm,false,BigFloat}

    # value is concretely typed and declared at the same element type as the
    # plain state, so nothing downstream can tell the states apart
    st = fold(Sum(:x), [1.5])
    @test @inferred(CausalFrames.value(st)) === (x_sum = 1.5,)

    # error-correcting summation: naive folding gives 0.0 here
    cancel = [1.0, 1e100, 1.0, -1e100]
    @test CausalFrames.value(fold(Sum(:x), cancel)) === (x_sum = 2.0,)
    df = DataFrame(
        load(Context(0, 9),
            chunks(DataFrame(time = 1:4, x = cancel)) |> summarize(Sum(:x))),
    )
    @test eltype(df.x_sum) == Float64 && only(df.x_sum) == 2.0

    # a NaN reports NaN (Base.sum semantics) but is counted, not folded in,
    # so downdating it recovers the finite sum exactly
    st = fold(Sum(:x), [1.5, NaN, 2.5])
    @test isnan(CausalFrames.value(st).x_sum)
    CausalFrames.downdate!(st, (; x = NaN))
    @test CausalFrames.value(st) === (x_sum = 4.0,)

    # infinities are counted by sign and reconstructed by IEEE rules; the
    # naive inverse would leave Inf - Inf = NaN behind forever
    st = fold(Sum(:x), [Inf, 1.0])
    @test CausalFrames.value(st).x_sum == Inf
    CausalFrames.update!(st, (; x = -Inf))
    @test isnan(CausalFrames.value(st).x_sum)
    CausalFrames.downdate!(st, (; x = Inf))
    @test CausalFrames.value(st).x_sum == -Inf
    CausalFrames.downdate!(st, (; x = -Inf))
    @test CausalFrames.value(st) === (x_sum = 1.0,)

    # the classified term is the folded one: after the power for SumPower
    # (NaN^0 == Inf^0 == 1.0 are finite terms), the per-row product for
    # DotProduct (Inf * 0.0 is a NaN term)
    @test CausalFrames.value(fold(SumPower(:x, 0), [NaN, Inf])) ===
          (x_sumpower_0 = 2.0,)
    st = fold(SumPower(:x, 2), [3.0, NaN])
    @test isnan(CausalFrames.value(st).x_sumpower_2)
    CausalFrames.downdate!(st, (; x = NaN))
    @test CausalFrames.value(st) === (x_sumpower_2 = 9.0,)
    st = CausalFrames.fresh(DotProduct(:x, :y), ftypes)
    CausalFrames.update!(st, (x = 2.0, y = 3.0))
    CausalFrames.update!(st, (x = Inf, y = 0.0))
    @test isnan(CausalFrames.value(st).x_y_dotproduct)
    CausalFrames.downdate!(st, (x = Inf, y = 0.0))
    @test CausalFrames.value(st) === (x_y_dotproduct = 6.0,)

    # combine! carries the compensation and the counts: every split of the
    # cancellation rows still gives the exact total, dest may alias, and a
    # fresh state is an identity
    whole = CausalFrames.value(fold(Sum(:x), cancel))
    for k in 0:length(cancel)
        a = fold(Sum(:x), cancel[1:k])
        b = fold(Sum(:x), cancel[(k+1):end])
        dest = CausalFrames.fresh(a)
        CausalFrames.combine!(dest, a, b)
        @test CausalFrames.value(dest) === whole
    end
    a = fold(Sum(:x), cancel[1:2])
    CausalFrames.combine!(a, a, fold(Sum(:x), cancel[3:4]))
    @test CausalFrames.value(a) === whole
    nanful = fold(Sum(:x), [1.0, NaN])
    dest = CausalFrames.fresh(nanful)
    CausalFrames.combine!(dest, nanful, fold(Sum(:x), [2.0]))
    @test isnan(CausalFrames.value(dest).x_sum)
    CausalFrames.downdate!(dest, (; x = NaN))
    @test CausalFrames.value(dest) === (x_sum = 3.0,)

    # widening carries the state across representation changes: plain Int
    # promotes into the compensated state, a compensated Float32 promotes to
    # a compensated Float64 keeping its compensation and counts, and a
    # missing-permitting promotion moves into the compensated *counting* state
    # (still invertible) keeping its compensation and nonfinite counts
    ist = CausalFrames.fresh(Sum(:x), (time = Int64, x = Int64))
    CausalFrames.update!(ist, (; x = 5))
    wst = CausalFrames.widenstate(ist, (time = Int64, x = Float64))
    @test wst isa CausalFrames.AccumState{:x_sum,Float64,xterm,false,Comp64}
    @test CausalFrames.value(wst) === (x_sum = 5.0,)

    f32 = CausalFrames.fresh(Sum(:x), (time = Int64, x = Float32))
    foreach(x -> CausalFrames.update!(f32, (; x)),
        Float32[1.0f10, 1.0f0, NaN32, Inf32])
    f64 = CausalFrames.widenstate(f32, (time = Int64, x = Float64))
    @test f64 isa CausalFrames.AccumState{:x_sum,Float64,xterm,false,Comp64}
    @test f64.acc.comp == Float64(f32.acc.comp) && f64.acc.comp != 0.0
    @test f64.acc.nans == 1 && f64.acc.posinf == 1
    CausalFrames.downdate!(f64, (; x = NaN))
    CausalFrames.downdate!(f64, (; x = Inf))
    @test CausalFrames.value(f64) === (x_sum = Float64(1.0f10) + 1.0,)

    nst = fold(Sum(:x), [1.0, NaN])
    mst = CausalFrames.widenstate(nst, (time = Int64, x = Union{Missing,Float64}))
    @test mst isa CausalFrames.AccumState{:x_sum,Float64,xterm,true,Comp64}
    @test mst.acc.nans == 1 && mst.missings == 0
    @test isnan(CausalFrames.value(mst).x_sum)
    @test CausalFrames.value(mst) isa NamedTuple{(:x_sum,),Tuple{Union{Missing,Float64}}}
    @test CausalFrames.isinvertible(mst)
    @test CausalFrames.isinvertible(nst)
end

@testset "missing counting summation" begin
    # A missing input term is counted, not folded in — the same trick the
    # compensated state uses for NaN/±Inf — so the accumulator stays invertible
    # and a rolling window recovers once a missing row leaves it. The
    # accumulation lives at the non-missing type; only value's return is
    # Union{Missing,_}.
    mint = (time = Int64, x = Union{Missing,Int}, y = Union{Missing,Int})
    mflt = (time = Int64, x = Union{Missing,Float64})

    function foldm(s, types, xs)
        st = CausalFrames.fresh(s, types)
        foreach(x -> CausalFrames.update!(st, (; x)), xs)
        return st
    end

    # value is missing iff a missing term is live, at the static Union type
    # (so it compares by isequal, not ===, against a concrete-typed literal)
    st = foldm(Sum(:x), mint, [1, missing, 3])
    @test CausalFrames.value(st) isa NamedTuple{(:x_sum,),Tuple{Union{Missing,Int}}}
    @test isequal(CausalFrames.value(st), (x_sum = missing,))
    CausalFrames.downdate!(st, (; x = missing))
    @test CausalFrames.value(st).x_sum === 4    # recovered exactly

    # the count balances across several missing terms
    st = foldm(Sum(:x), mint, [1, missing, missing, 4])
    @test st.missings == 2 && st.acc == 5
    @test isequal(CausalFrames.value(st), (x_sum = missing,))
    CausalFrames.downdate!(st, (; x = missing))
    @test isequal(CausalFrames.value(st), (x_sum = missing,))   # one still live
    CausalFrames.downdate!(st, (; x = missing))
    @test isequal(CausalFrames.value(st), (x_sum = 5,))

    # missing dominates NaN in the counting float state, and both counts
    # subtract away independently
    st = foldm(Sum(:x), mflt, [1.0, NaN, missing, 4.0])
    @test st.acc.nans == 1 && st.missings == 1
    @test isequal(CausalFrames.value(st), (x_sum = missing,))
    CausalFrames.downdate!(st, (; x = missing))
    @test isnan(CausalFrames.value(st).x_sum)                   # NaN still live
    CausalFrames.downdate!(st, (; x = NaN))
    @test isequal(CausalFrames.value(st), (x_sum = 5.0,))

    # combine! adds the counts and reads before writing, so dest may alias and
    # a fresh state is the identity (this is what the tree mode relies on)
    a = foldm(Sum(:x), mint, [1, missing])
    b = foldm(Sum(:x), mint, [missing, 4])
    dest = CausalFrames.fresh(a)
    CausalFrames.combine!(dest, a, b)
    @test dest.missings == 2 && dest.acc == 5
    @test isequal(CausalFrames.value(dest), (x_sum = missing,))
    CausalFrames.combine!(a, a, foldm(Sum(:x), mint, [10]))     # dest aliases a
    @test a.missings == 1 && a.acc == 11

    # DotProduct counts a term missing when either operand is
    st = CausalFrames.fresh(DotProduct(:x, :y), mint)
    CausalFrames.update!(st, (x = 2, y = 3))
    CausalFrames.update!(st, (x = missing, y = 5))
    @test st.missings == 1
    @test isequal(CausalFrames.value(st), (x_y_dotproduct = missing,))
    CausalFrames.downdate!(st, (x = missing, y = 5))
    @test isequal(CausalFrames.value(st), (x_y_dotproduct = 6,))

    # update!/downdate! on the counting states allocate nothing on the hot path
    # (measured behind a function barrier, as the folding kernels always run)
    function updalloc(st, row)
        CausalFrames.update!(st, row)                # warm up this specialization
        return @allocated CausalFrames.update!(st, row)
    end
    st = foldm(Sum(:x), mint, Int[])
    @test updalloc(st, (x = 3,)) == 0
    @test updalloc(st, (x = missing,)) == 0
end

# A pseudo-random walk over a sliding window: each step admits a row or evicts
# the oldest, and the check runs after every step. Admissions win two times in
# three, so the window grows, drains and refills.
function windowwalk(check, vals, nsteps; seed)
    ops = lcgsequence(seed, nsteps, 3)
    picks = lcgsequence(seed + 1, nsteps, length(vals))
    live = NamedTuple[]
    for i in 1:nsteps
        if ops[i] == 0 && !isempty(live)
            check(:evict, popfirst!(live), live)
        else
            row = (time = i, x = vals[picks[i]+1])
            push!(live, row)
            check(:admit, row, live)
        end
    end
    return nothing
end

@testset "windowed states" begin
    # The deque behind the windowed Min/Max/First/Last, and CountDistinct's
    # counts, against the definition: a fresh ordinary state folded over the
    # live window. The pools carry what a selection can trip over — ties,
    # ±0.0, NaN and missing (min and max both propagate the last two) — and
    # the deque must agree under isequal, not just ==.
    pools = [
        Union{Missing,Float64}[1.0, 2.0, 2.0, -0.0, 0.0, NaN, missing, 3.0, -1.0],
        [3, 1, 4, 1, 5, 9, 2, 6],
        ["b", "a", "c", "a"],
    ]
    for vals in pools, s in (Min(:x), Max(:x), First(:x), Last(:x),
            CountDistinct(:x))

        intypes = (time = Int, x = eltype(vals))
        ws = CausalFrames.freshwindowed(s, intypes)
        @test !(ws isa typeof(CausalFrames.fresh(s, intypes)))
        # one assertion per walk, naming every step that disagreed
        bad = Tuple{Symbol,Int}[]
        windowwalk(vals, 400; seed = 7) do op, row, live
            op === :admit ? CausalFrames.update!(ws, row) :
            CausalFrames.downdate!(ws, row)
            isempty(live) && return
            ref = CausalFrames.fresh(s, intypes)
            foreach(r -> CausalFrames.update!(ref, r), live)
            isequal(CausalFrames.value(ws), CausalFrames.value(ref)) ||
                push!(bad, (op, row.time))
        end
        @test isempty(bad)
        # zeroing gives a state indistinguishable from a fresh one
        ws = CausalFrames.fresh!(ws)
        ref = CausalFrames.fresh(s, intypes)
        CausalFrames.update!(ws, (time = 0, x = vals[2]))
        CausalFrames.update!(ref, (time = 0, x = vals[2]))
        @test isequal(CausalFrames.value(ws), CausalFrames.value(ref))
    end

    # Min/Max/First slide the deque; Last only needs a count and the newest value
    @test CausalFrames.freshwindowed(Last(:x), (time = Int, x = Float64)) isa
          CausalFrames.WindowLastState
    @test CausalFrames.freshwindowed(First(:x), (time = Int, x = Float64)) isa
          CausalFrames.WindowTrackState

    # Every summarizer without its own windowed state gets its ordinary one.
    intypes = (time = Int, x = Int)
    @test CausalFrames.freshwindowed(Sum(:x), intypes) isa
          typeof(CausalFrames.fresh(Sum(:x), intypes))

    # A steady window slides without allocating: the deque reclaims its dead
    # front slots in place, and a retired count leaves the Dict's slots behind.
    slrow(t) = (time = t, x = Float64(mod(t, 17)))
    function slidealloc(ws, from, n)
        for t in from:(from+n-1)
            CausalFrames.update!(ws, slrow(t))
            CausalFrames.downdate!(ws, slrow(t - 50))
        end
        return nothing
    end
    for s in (Min(:x), Max(:x), First(:x), Last(:x), CountDistinct(:x))
        ws = CausalFrames.freshwindowed(s, (time = Int, x = Float64))
        foreach(t -> CausalFrames.update!(ws, slrow(t)), -49:0)
        slidealloc(ws, 1, 1000)
        @test (@allocated slidealloc(ws, 1001, 1000)) == 0
    end
end

@testset "age-weighted sum" begin
    s = AgeWeightedSum(:x)
    @test CausalFrames.emptyvalue(s) === (x_ageweightedsum = 0,)

    # The definition, Σ k·y with k the row's age and the newest weighing 0 — so
    # a nonfinite newest value contributes nothing — and missing anywhere
    # poisoning the whole.
    function naive(live)
        any(r -> ismissing(r.x), live) && return missing
        n = length(live)
        acc = 0 * live[1].x
        for (j, r) in enumerate(live)
            k = n - j
            k == 0 || (acc += k * r.x)
        end
        return acc
    end
    agree(a, b) = isequal(a, b) || (a isa AbstractFloat && isapprox(a, b))
    fold(intypes, rows) = foldl((st, r) -> (CausalFrames.update!(st, r); st),
        rows; init = CausalFrames.fresh(s, intypes))

    pools = [
        [3, -1, 4, 1, -5, 9, 2, 6],
        [0.1, -2.5, 3.0, 1e8, 0.3, Inf, -Inf, NaN, 7.25],
        Union{Missing,Int}[2, missing, 5, -3],
        Union{Missing,Float64}[0.5, missing, Inf, 1.5, NaN],
    ]
    for vals in pools
        intypes = (time = Int, x = eltype(vals))
        st = CausalFrames.fresh(s, intypes)
        # one assertion per walk and check, naming every step that disagreed
        badvalue = Tuple{Symbol,Int}[]
        badsplit = Tuple{Symbol,Int,Int}[]
        windowwalk(vals, 500; seed = 3) do op, row, live
            op === :admit ? CausalFrames.update!(st, row) :
            CausalFrames.downdate!(st, row)
            isempty(live) && return
            agree(CausalFrames.value(st).x_ageweightedsum, naive(live)) ||
                push!(badvalue, (op, row.time))
            # combine! over every split point of the live window equals the fold
            length(live) > 6 && return
            for k in 0:length(live)
                dest = CausalFrames.fresh(st)
                CausalFrames.combine!(dest, fold(intypes, live[1:k]),
                    fold(intypes, live[(k+1):end]))
                agree(CausalFrames.value(dest).x_ageweightedsum, naive(live)) ||
                    push!(badsplit, (op, row.time, k))
            end
        end
        @test isempty(badvalue)
        @test isempty(badsplit)
    end

    # The newest row weighs 0: a nonfinite value there waits for a later row.
    ft = (time = Int, x = Float64)
    st = fold(ft, [(time = 1, x = 1.0), (time = 2, x = 2.0), (time = 3, x = Inf)])
    @test CausalFrames.value(st) === (x_ageweightedsum = 4.0,)
    CausalFrames.update!(st, (time = 4, x = 5.0))
    @test CausalFrames.value(st) === (x_ageweightedsum = Inf,)
    CausalFrames.update!(st, (time = 5, x = NaN))
    @test CausalFrames.value(st).x_ageweightedsum === Inf
    CausalFrames.update!(st, (time = 6, x = 0.0))
    @test isnan(CausalFrames.value(st).x_ageweightedsum)
    for x in (1.0, 2.0, Inf, 5.0, NaN)     # every row but the last leaves
        CausalFrames.downdate!(st, (time = 0, x = x))
    end
    @test CausalFrames.value(st) === (x_ageweightedsum = 0.0,)

    # The element type follows Sum, and a missing-admitting column's value
    # admits missing.
    @test CausalFrames.value(fold((time = Int, x = Int32),
        [(time = 1, x = Int32(1))])) === (x_ageweightedsum = 0,)
    @test fieldtype(
        typeof(
            CausalFrames.value(
                fold((time = Int,
                    x = Union{Missing,Float64}), [(time = 1, x = 1.0)]),
            ),
        ), 1) ==
          Union{Missing,Float64}

    # Widening carries (n, S1, S2): Int to Float64, then to Missing-admitting,
    # after which the older rows still downdate exactly.
    rows = [(time = 1, x = 3), (time = 2, x = 1), (time = 3, x = 4)]
    st = fold((time = Int, x = Int), rows)
    st = CausalFrames.widenstate(st, (time = Int, x = Float64))
    @test st isa CausalFrames.CompensatedAgeSumState
    CausalFrames.update!(st, (time = 4, x = 0.5))
    st = CausalFrames.widenstate(st, (time = Int, x = Union{Missing,Float64}))
    CausalFrames.update!(st, (time = 5, x = missing))
    @test ismissing(CausalFrames.value(st).x_ageweightedsum)
    CausalFrames.downdate!(st, rows[1])
    @test ismissing(CausalFrames.value(st).x_ageweightedsum)
    st2 = CausalFrames.widenstate(fold((time = Int, x = Int), rows),
        (time = Int, x = Union{Missing,Int}))
    @test st2 isa CausalFrames.AgeSumState
    @test CausalFrames.value(st2) == (x_ageweightedsum = 2 * 3 + 1 * 1,)

    # update!/downdate! allocate nothing on either representation (a lone row
    # in, then out: the oldest row is the only one)
    function agealloc(st, row)
        CausalFrames.update!(st, row)
        CausalFrames.downdate!(st, row)
        return @allocated begin
            CausalFrames.update!(st, row)
            CausalFrames.downdate!(st, row)
        end
    end
    @test agealloc(CausalFrames.fresh(s, (time = Int, x = Int)),
        (time = 9, x = 2)) == 0
    @test agealloc(CausalFrames.fresh(s, (time = Int, x = Union{Missing,Float64})),
        (time = 9, x = 2.0)) == 0
end

@testset "order statistics" begin
    SV = CausalFrames.SortedValues
    fold(s, intypes, xs) = foldl(xs; init = CausalFrames.fresh(s, intypes)) do st, x
        CausalFrames.update!(st, (time = 0, x = x))
        st
    end
    summarized(xs, ss) = DataFrame(
        load(Context(0, 99),
            readtable(DataFrame(time = eachindex(xs), x = xs)) |> summarize(ss)),
    )

    # names carry the percentage, 0.07 included (100 * 0.07 is not 7), in the
    # order the probabilities were given
    @test CausalFrames.emptyvalue(Quantile(:x, 0.5)) === (x_quantile_50 = missing,)
    @test keys(CausalFrames.emptyvalue(Quantile(:x, [0.9, 0.025, 0.07, 0, 1]))) ==
          (:x_quantile_90, :x_quantile_2_5, :x_quantile_7, :x_quantile_0,
        :x_quantile_100)
    @test CausalFrames.emptyvalue(Median(:x)) === (x_median = missing,)
    @test CausalFrames.emptyvalue(PercentRank(:x)) === (x_percentrank = missing,)
    @test CausalFrames.emptyvalue(SV(:x)) === (x_sortedvalues = missing,)
    @test all(s -> s isa GroupSummarizer,
        [SV(:x), Quantile(:x, 0.5), Median(:x), PercentRank(:x)])
    @test :SortedValues ∉ names(CausalFrames)

    @test_throws ArgumentError("Quantile requires at least one probability") Quantile(
        :x, Float64[])
    @test_throws ArgumentError("Quantile probabilities must be unique, got (0.5, 0.5)") Quantile(
        :x, [0.5, 0.5])
    for p in (-0.1, 1.5, NaN)
        @test_throws ArgumentError("Quantile probability must be in [0, 1], got $p") Quantile(
            :x, p)
    end
    @test_throws ArgumentError(
        "Quantile interpolation must be :linear or :nearestrank, got :cubic") Quantile(
        :x, 0.5; interpolation = :cubic)
    @test_throws ArgumentError Quantile(:x, [0.1, 0.1 + 1e-15])
    # one probability requested twice is the ordinary name clash
    @test_throws ArgumentError summarized([1, 2],
        [Quantile(:x, [0.25, 0.5]), Quantile(:x, 0.5)])

    # every order statistic of a column shares the one accumulator
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Quantile(:x, 0.25), Median(:x),
            Quantile(:x, 0.9; interpolation = :nearestrank), PercentRank(:x)]),
        Symbol[])
    @test count(s -> s isa SV, protos) == 1
    @test requested == (:x_quantile_25, :x_median, :x_quantile_90, :x_percentrank)

    # Against the definitions over every prefix of a sequence with ties: the
    # linear rule is Statistics.quantile's (to rounding: its last bit moved
    # between versions), the nearest rank is TA-Lib's PERCENTILE in exact
    # integer arithmetic, and the percent rank is TA-Lib's PERCENTRANK / 100
    # over the rows before the newest.
    xs = map(v -> v - 5, lcgsequence(11, 60, 11))
    Ps = (0, 7, 25, 50, 90, 100)
    ps = map(P -> P / 100, Ps)
    qnames = keys(CausalFrames.emptyvalue(Quantile(:x, ps)))
    running(ss) = DataFrame(
        load(Context(0, 99),
            readtable(DataFrame(time = eachindex(xs), x = xs)) |> addsummarycolumns(ss),
        ),
    )
    lin = running([Quantile(:x, ps), Median(:x), PercentRank(:x)])
    near = running(Quantile(:x, ps; interpolation = :nearestrank))
    bad = Int[]
    for n in eachindex(xs)
        w = xs[1:n]
        s = sort(w)
        ok =
            all(i -> lin[n, qnames[i]] ≈ Statistics.quantile(w, ps[i]), eachindex(ps)) &&
            lin.x_median[n] ≈ Statistics.median(w) &&
            isequal(lin.x_percentrank[n], count(<(w[end]), w[1:(end-1)]) / (n - 1)) &&
            all(i -> near[n, qnames[i]] == s[clamp(cld(Ps[i] * n, 100), 1, n)],
                eachindex(Ps))
        ok || push!(bad, n)
    end
    @test isempty(bad)
    # 0.07 over 100 rows is the 7th value, where ceil(0.07 * 100) would be 8
    st = fold(SV(:x), (time = Int, x = Int), 1:100)
    q = CausalFrames.fresh(Quantile(:x, 0.07; interpolation = :nearestrank),
        (time = Int, x = Int))
    @test CausalFrames.value(q, CausalFrames.value(st)) === (x_quantile_7 = 7,)
    # and the other way: nextfloat(1/3) * 3 rounds down to 1.0, but 1/3 < p, so
    # the rank is 2
    near(p) = only(summarized([10, 20, 30],
        Quantile(:x, p; interpolation = :nearestrank))[!, 2])
    @test (near(1 / 3), near(nextfloat(1 / 3))) == (10, 20)

    # element types: linear interpolation floats, the nearest rank keeps the
    # column's type, the percent rank is a Float64 fraction; a Missing-admitting
    # column admits missing
    for (T, lin, near) in ((Int, Float64, Int), (Float32, Float64, Float32),
        (Union{Missing,Int}, Union{Missing,Float64}, Union{Missing,Int}))
        df = summarized(
            T[3, 1, 2],
            [Quantile(:x, 0.5), PercentRank(:x),
                Quantile(:x, 0.25; interpolation = :nearestrank)],
        )
        @test eltype(df.x_quantile_50) == lin
        @test eltype(df.x_quantile_25) == near
        @test eltype(df.x_percentrank) ==
              (Missing <: T ? Union{Missing,Float64} : Float64)
        @test isequal(Vector(df[1, 2:end]), [2.0, 0.5, 1])
    end
    # nearest rank needs only an order
    df = summarized(["b", "c", "a"], Quantile(:x, [0, 1]; interpolation = :nearestrank))
    @test (only(df.x_quantile_0), only(df.x_quantile_100)) == ("a", "c")

    # a missing gives missing and a NaN gives NaN, everywhere; one row ranks NaN
    ss = [Quantile(:x, [0.1, 0.9]), Median(:x), PercentRank(:x),
        Quantile(:x, 0.5; interpolation = :nearestrank)]
    @test all(ismissing, summarized([1.0, missing, 3.0], ss)[1, 2:end])
    @test all(isnan, summarized([1.0, NaN, 3.0], ss)[1, 2:end])
    @test all(isnan, summarized([1.0, 2.0, NaN], ss)[1, 2:end])
    @test isnan(only(summarized([4], PercentRank(:x)).x_percentrank))
    # no rows: the empty values
    df = DataFrame(load(Context(0, 9), emptyframe() |> summarize(ss)))
    @test all(ismissing, df[1, 2:end])

    # The accumulator slid over a live window against a fresh fold of it: ties
    # (downdate! removes exactly one copy), ±0.0 kept apart, NaN and missing
    # counted, strings ordered.
    pools = [
        Union{Missing,Float64}[1.0, 2.0, 2.0, -0.0, 0.0, NaN, missing, 3.0, -1.0],
        [3, 1, 4, 1, 5, 9, 2, 6, 1],
        ["b", "a", "c", "a"],
    ]
    snapshot(st) = (copy(st.vals), st.nans, st.missings)
    for vals in pools
        intypes = (time = Int, x = eltype(vals))
        ws = CausalFrames.freshwindowed(SV(:x), intypes)
        bad = Tuple{Symbol,Int}[]
        windowwalk(vals, 400; seed = 5) do op, row, live
            op === :admit ? CausalFrames.update!(ws, row) :
            CausalFrames.downdate!(ws, row)
            ref = fold(SV(:x), intypes, [r.x for r in live])
            isequal(snapshot(ws), snapshot(ref)) || push!(bad, (op, row.time))
        end
        @test isempty(bad)
        @test isequal(snapshot(CausalFrames.fresh!(ws)),
            snapshot(CausalFrames.fresh(SV(:x), intypes)))

        # combine! is the fold of both, whichever state it overwrites
        a1, b1 = vals[1:(end÷2)], vals[(end÷2+1):end]
        whole = snapshot(fold(SV(:x), intypes, vals))
        for alias in (:none, :a, :b)
            a, b = fold(SV(:x), intypes, a1), fold(SV(:x), intypes, b1)
            dest = alias === :a ? a : alias === :b ? b :
                                      fold(SV(:x), intypes, vals[1:2])
            CausalFrames.combine!(dest, a, b)
            @test isequal(snapshot(dest), whole)
        end
    end

    # widening keeps the order and the counts
    st = fold(SV(:x), (time = Int, x = Int), [3, 1, 2])
    st = CausalFrames.widenstate(st, (time = Int, x = Union{Missing,Float64}))
    @test st isa CausalFrames.SortedState{:x,:x_sortedvalues,Float64,true}
    CausalFrames.update!(st, (time = 4, x = missing))
    CausalFrames.update!(st, (time = 5, x = 1.5))
    @test isequal(snapshot(st), ([1.0, 1.5, 2.0, 3.0], 0, 1))

    # a steady window slides, and emits, without allocating
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers(
            [Quantile(:x, [0.1, 0.5]), Median(:x), PercentRank(:x),
            Quantile(:x, 0.9; interpolation = :nearestrank)]), Symbol[])
    intypes = (time = Int, x = Union{Missing,Float64})
    states = map(s -> CausalFrames.freshwindowed(s, intypes), protos)
    outs = Val(requested)
    slrow(t) = (time = t, x = Float64(mod(t * 7, 17)))
    function slide(states, from, n)
        for t in from:(from+n-1)
            CausalFrames.updateall!(states, slrow(t))
            CausalFrames.downdateall!(states, slrow(t - 50))
            CausalFrames.summaryvalues(states, outs)
        end
        return nothing
    end
    foreach(t -> CausalFrames.updateall!(states, slrow(t)), -49:0)
    slide(states, 1, 1000)
    @test (@allocated slide(states, 1001, 1000)) == 0
end
