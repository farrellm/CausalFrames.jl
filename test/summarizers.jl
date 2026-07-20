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
            Covariance(:x, :y), Correlation(:x, :y)])
    @test all(s -> s isa MonoidSummarizer && !(s isa GroupSummarizer),
        [Product(:x), Min(:x), Max(:x), First(:x), Last(:x), MinMax(:x)])
    @test !(Opaque(Sum(:x)) isa MonoidSummarizer)

    monoids = [Count(), Sum(:x), SumPower(:x, 2), DotProduct(:x, :y),
        Product(:x), Min(:x), Max(:x), First(:x), Last(:x), MinMax(:x)]

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

    # invertibility is a property of the realized accumulator type: a
    # missing-permitting accumulator absorbs and cannot be downdated
    @test CausalFrames.isinvertible(CausalFrames.fresh(Sum(:x), intypes))
    @test CausalFrames.isinvertible(CausalFrames.fresh(Count(), intypes))
    mintypes = (time = Int64, x = Union{Missing,Int}, y = Union{Missing,Int})
    for s in [Sum(:x), SumPower(:x, 2), DotProduct(:x, :y)]
        @test !CausalFrames.isinvertible(CausalFrames.fresh(s, mintypes))
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

    # float accumulators get the compensated state; missing-permitting and
    # BigFloat accumulators keep the plain one
    @test CausalFrames.fresh(Sum(:x), ftypes) isa
          CausalFrames.CompensatedAccumState{:x_sum,Float64,
        CausalFrames.ColumnTerm{:x}}
    @test CausalFrames.fresh(Sum(:x), (time = Int64, x = Union{Missing,Float64})) isa
          CausalFrames.AccumState
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
    # missing-permitting promotion falls back to the plain (absorbing) state
    # holding the reconstructed value
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
    @test mst isa CausalFrames.AccumState{:x_sum,Union{Missing,Float64},
        CausalFrames.ColumnTerm{:x}}
    @test isnan(CausalFrames.value(mst).x_sum)
    @test !CausalFrames.isinvertible(mst)
    @test CausalFrames.isinvertible(nst)
end
