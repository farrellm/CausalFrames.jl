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

    # a source may infer a column's type per chunk, so the state widens
    # across the boundary rather than forcing the first chunk's type
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

    # Covariance(:x, :x) is Variance(:x)
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4], [Covariance(:x, :x)])
    @test only(df.x_x_covariance) == 1.0

    # element types: Product/DotProduct widen like Sum; the statistical
    # dependents divide integers to Float64
    @test eltype(df.x_x_covariance) == Float64
    df = stats(Int32[3, 1, 2], Int32[2, 5, 4],
        [Product(:x), DotProduct(:x, :y), Mean(:x), Variance(:x),
            Std(:x), Covariance(:x, :y)])
    @test eltype(df.x_product) == Int64
    @test eltype(df.x_y_dotproduct) == Int64
    @test all(
        eltype(df[!, c]) == Float64
        for c in [:x_mean, :x_variance, :x_std, :x_y_covariance]
    )

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

    # the sharing claim, tested rather than assumed: two regressions over the
    # same predictors fold one accumulator per cross product between them
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
    # Covariance does not canonicalize its own dependency, so the reversed
    # argument order folds a second accumulator for the same quantity
    protos, _ = CausalFrames.prototypes(
        Summarizer[LinearRegression(:x, :y; name = :m1), Covariance(:y, :x)],
        Symbol[])
    folded = [
        only(keys(CausalFrames.emptyvalue(s))) for s in protos
        if length(keys(CausalFrames.emptyvalue(s))) == 1
    ]
    @test :x_y_dotproduct in folded && :y_x_dotproduct in folded

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

    # the hierarchy: the accumulators and the dependent summarizers are
    # groups; Product and the trackers are monoids only; a plain Summarizer
    # is neither
    @test all(s -> s isa GroupSummarizer,
        [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
            Moment(:x, 2), Mean(:x), Variance(:x), Std(:x),
            Covariance(:x, :y), Correlation(:x, :y),
            LinearRegression(:x, :y), LinearRegression([:x, :y], :y)])
    @test all(s -> s isa MonoidSummarizer && !(s isa GroupSummarizer),
        [Product(:x), Min(:x), Max(:x), First(:x), Last(:x), MinMax(:x)])
    @test !(Opaque(Sum(:x)) isa MonoidSummarizer)

    monoids = [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        Product(:x), Min(:x), Max(:x), First(:x), Last(:x), MinMax(:x)]

    # fresh! must be indistinguishable from fresh: the transforms zero and
    # reuse state tuples per cycle, per interval and per window query, so a
    # state that does not fully reset leaks one emission's values into the
    # next. MinMax is in the list without implementing fresh!, which is what
    # exercises the `fresh(st)` default a custom summarizer inherits.
    selfcontained = [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        Product(:x), Min(:x), Max(:x), First(:x), Last(:x), MinMax(:x),
        Opaque(Sum(:x))]
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

    # Zeroing a built-in state is pure field writes, so it allocates nothing —
    # the property every reuse path depends on.
    for s in [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        Product(:x), Min(:x), Max(:x), First(:x), Last(:x)]
        st = fold(s, rows)
        CausalFrames.fresh!(st)
        @test (@allocated CausalFrames.fresh!(st)) == 0
    end

    # And over a whole tuple the cost does not grow with the number of resets.
    # Asserted as a slope rather than as `@allocated(one call) == 0` because a
    # lone `@allocated` puts a function boundary around the call, and the
    # returned tuple has to be materialized to cross it — which Julia 1.10
    # does even though it elides the tuple once `freshall!` is inlined into a
    # folding loop, which is the only way the package calls it.
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
    @test CausalFrames.fresh(Sum(:x), ftypes) isa
          CausalFrames.CompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = Union{Missing,Float64})) isa
          CausalFrames.OptionalCompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = Union{Missing,Int})) isa
          CausalFrames.OptionalAccumState{:x_sum,Int,CausalFrames.ColumnTerm{:x}}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = BigFloat)) isa
          CausalFrames.AccumState

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
    @test wst isa CausalFrames.CompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
    @test CausalFrames.value(wst) === (x_sum = 5.0,)

    f32 = CausalFrames.fresh(Sum(:x), (time = Int64, x = Float32))
    foreach(x -> CausalFrames.update!(f32, (; x)),
        Float32[1.0f10, 1.0f0, NaN32, Inf32])
    f64 = CausalFrames.widenstate(f32, (time = Int64, x = Float64))
    @test f64 isa CausalFrames.CompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
    @test f64.acc.comp == Float64(f32.acc.comp) && f64.acc.comp != 0.0
    @test f64.acc.nans == 1 && f64.acc.posinf == 1
    CausalFrames.downdate!(f64, (; x = NaN))
    CausalFrames.downdate!(f64, (; x = Inf))
    @test CausalFrames.value(f64) === (x_sum = Float64(1.0f10) + 1.0,)

    nst = fold(Sum(:x), [1.0, NaN])
    mst = CausalFrames.widenstate(nst, (time = Int64, x = Union{Missing,Float64}))
    @test mst isa CausalFrames.OptionalCompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
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
    @test isequal(CausalFrames.value(st), (x_sum = 4,))     # recovered exactly
    @test CausalFrames.value(st).x_sum === 4

    # the count balances across several missing terms
    st = foldm(Sum(:x), mint, [1, missing, missing, 4])
    @test st.missings == 2 && st.total == 5
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
    @test dest.missings == 2 && dest.total == 5
    @test isequal(CausalFrames.value(dest), (x_sum = missing,))
    CausalFrames.combine!(a, a, foldm(Sum(:x), mint, [10]))     # dest aliases a
    @test a.missings == 1 && a.total == 11

    # DotProduct counts a term missing when either operand is
    st = CausalFrames.fresh(DotProduct(:x, :y), mint)
    CausalFrames.update!(st, (x = 2, y = 3))
    CausalFrames.update!(st, (x = missing, y = 5))
    @test st.missings == 1
    @test isequal(CausalFrames.value(st), (x_y_dotproduct = missing,))
    CausalFrames.downdate!(st, (x = missing, y = 5))
    @test isequal(CausalFrames.value(st), (x_y_dotproduct = 6,))

    # dependent summarizers inherit missing through the shared value NamedTuple,
    # and keep a Union{Missing,_} eltype rather than collapsing to Missing
    p = CausalPipeline(ctx -> [DataFrame(time = [1, 2, 3], x = [2, missing, 4])])
    df = DataFrame(load(Context(0, 9), p |> summarize([Mean(:x), Variance(:x)])))
    @test eltype(df.x_mean) == Union{Missing,Float64}
    @test eltype(df.x_variance) == Union{Missing,Float64}
    @test ismissing(only(df.x_mean)) && ismissing(only(df.x_variance))

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
