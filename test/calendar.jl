# Calendar periods (Month, Quarter, Year) as tolerances, look-backs, clock
# intervals and shifts. They have no fixed length, so every window is measured
# on the calendar — `s` is within `lb` of `t` when `s >= t - lb` — and a clock
# is anchored to its start. Included after models.jl, whose ToyOLS it reuses.
@testset "calendar periods" begin
    dt(m, d) = DateTime(2026, m, d)
    ctx = Context(dt(1, 1), dt(12, 1))
    frame(p, c = ctx) = DataFrame(load(c, p))

    @testset "clock is anchored to start" begin
        ticks(c, i; kw...) = frame(clock(i; kw...), c).time
        c = Context(dt(1, 31), dt(6, 1))
        want = [dt(1, 31), dt(2, 28), dt(3, 31), dt(4, 30), dt(5, 31)]
        @test ticks(c, Month(1)) == want
        @test ticks(c, Month(1); batchsize = 2) == want   # across chunks too
        @test ticks(Context(Date(2024, 2, 29), Date(2029, 1, 1)), Year(1)) ==
              [Date(2024, 2, 29), Date(2025, 2, 28), Date(2026, 2, 28),
            Date(2027, 2, 28), Date(2028, 2, 29)]
        @test ticks(Context(dt(1, 31), dt(12, 31)), Quarter(1)) ==
              [dt(1, 31), dt(4, 30), dt(7, 31), dt(10, 31)]
        # a fixed interval still accumulates, exactly as before
        @test frame(clock(0.1), Context(0.0, 0.35)).time == [0.0, 0.1, 0.2, 0.1 + 0.1 + 0.1]
        @test_throws ArgumentError load(ctx, clock(Month(0)))
        @test_throws ArgumentError load(ctx, clock(Month(-1)))
    end

    @testset "asofjoin tolerance" begin
        right = readtable(DataFrame(time = [dt(1, 31), dt(2, 15)], v = [1, 2]))
        left = readtable(DataFrame(time = [dt(2, 28), dt(3, 15), dt(3, 16)]))
        # Mar 15 reaches back exactly to Feb 15 (inclusive); Mar 16 only to Feb 16
        df = frame(left |> asofjoin(right; tolerance = Month(1)))
        @test isequal(df.v, [2, 2, missing])

        # the month-end clamp: Mar 31 - Month(1) is Feb 28
        r = readtable(DataFrame(time = [dt(2, 27), dt(2, 28)], v = [1, 2]))
        l = readtable(DataFrame(time = [dt(3, 31)]))
        @test frame(l |> asofjoin(r; tolerance = Month(1))).v == [2]
        r1 = readtable(DataFrame(time = [dt(2, 27)], v = [1]))
        @test isequal(frame(l |> asofjoin(r1; tolerance = Month(1))).v, [missing])

        # Year over Date time, across a leap day; the right side is read from
        # the widened context, a year before start
        r = readtable(DataFrame(time = [Date(2024, 2, 29)], v = [1]))
        l = readtable(DataFrame(time = [Date(2025, 2, 28), Date(2025, 3, 1)]))
        df = frame(l |> asofjoin(r; tolerance = Year(1)),
            Context(Date(2025, 1, 1), Date(2026, 1, 1)))
        @test isequal(df.v, [1, missing])

        @test_throws ArgumentError load(ctx,
            left |> asofjoin(right; tolerance = Month(-1)))
    end

    @testset "futurejoin tolerance" begin
        right = readtable(DataFrame(time = [dt(2, 28), dt(4, 1)], v = [1, 2]))
        left = readtable(DataFrame(time = [dt(1, 27), dt(1, 31), dt(3, 1)]))
        # Jan 27 reaches ahead to Feb 27; Jan 31 to Feb 28 (clamped); Mar 1 to Apr 1
        df = frame(left |> futurejoin(right; tolerance = Month(1)))
        @test isequal(df.v, [missing, 1, 2])
        @test_throws ArgumentError load(ctx,
            left |> futurejoin(right; tolerance = Month(-1)))
    end

    @testset "forwardfill tolerance" begin
        p = readtable(
            DataFrame(time = [dt(1, 31), dt(2, 28), dt(3, 1)],
                x = [1.0, missing, missing]),
        )
        # Feb 28 reaches back to Jan 28; Mar 1 only to Feb 1
        df = frame(p |> forwardfill(:x; tolerance = Month(1)))
        @test isequal(df.x, [1.0, 1.0, missing])
    end

    @testset "lag and lead shift on the calendar" begin
        p = readtable(DataFrame(time = [dt(1, 30), dt(2, 28), dt(3, 5)], v = [0, 1, 2]))
        lagged(c) = frame(p |> lag(Month(1)), c)
        # a row shifted before start is clipped (Feb 28 lands on Mar 28)
        df = lagged(Context(dt(3, 30), dt(6, 1)))
        @test df.time == [dt(4, 5)]
        @test df.v == [2]
        # and one read from past stop - Month(1) is not lost (Feb 28 -> Mar 28)
        df = lagged(Context(dt(2, 1), dt(3, 31)))
        @test df.time == [dt(2, 28), dt(3, 28)]
        @test df.v == [0, 1]
        # stateless, so split windows concatenate to the whole
        whole = lagged(ctx)
        @test whole.time == [dt(2, 28), dt(3, 28), dt(4, 5)]
        @test vcat(lagged(Context(dt(1, 1), dt(3, 28))),
            lagged(Context(dt(3, 28), dt(12, 1)))) == whole

        led(c) = frame(p |> lead(Month(1)), c)
        # Feb 28 - Month(1) is Jan 28: read from past stop + Month(1) = Feb 28
        df = led(Context(dt(1, 1), dt(1, 31)))
        @test df.time == [dt(1, 28)]
        @test df.v == [1]
        @test led(ctx).time == [dt(1, 28), dt(2, 5)]   # Jan 30 lands in 2025

        @test_throws ArgumentError load(ctx, p |> lag(Month(-1)))
        @test_throws ArgumentError load(ctx, p |> lead(Month(-1)))
    end

    # Uneven gaps of 0 to 8 days, two keys, spanning a few month ends.
    n = 60
    ts = dt(1, 1) .+ Day.(cumsum(lcgsequence(41, n, 9)))
    xs = lcgsequence(42, n, 100)
    ks = map(v -> ("a", "b")[v+1], lcgsequence(43, n, 2))
    data = readtable(DataFrame(time = ts, k = ks, x = xs))
    inwindow(t, s, lb) = t - lb <= s          # the calendar reading, oracle-side

    @testset "addrollingcolumns look-backs" begin
        c = Context(dt(2, 1), dt(12, 1))
        out = findall(>=(c.start), ts)
        windows = (m = Month(1), d = Day(10), q = Quarter(1))
        oracle(lb, i, keyed, f) = f([
            xs[j] for j in 1:n
            if ts[j] <= ts[i] && inwindow(ts[i], ts[j], lb) &&
                (!keyed || ks[j] == ks[i])
        ])
        # running (Sum), tree (Min) and re-fold (Opaque) paths
        for (ss, col, f) in ((Sum(:x), "x_sum", sum), (Min(:x), "x_min", minimum),
            (Opaque(Sum(:x)), "x_sum", sum))
            for keyed in (false, true)
                kw = keyed ? (; key = :k) : (;)
                df = frame(data |> addrollingcolumns(windows, ss; kw...), c)
                @test df.time == ts[out]
                for (w, lb) in pairs(windows)
                    @test df[!, "$(w)_$col"] == [oracle(lb, i, keyed, f) for i in out]
                end
            end
        end
    end

    @testset "summarizewindows look-backs" begin
        c = Context(dt(2, 1), dt(12, 1))
        clk = clock(Month(1))
        τs = frame(clk, c).time
        oracle(τ, lb, f, empty) = (v = [xs[j] for j in 1:n
                        if ts[j] < τ && inwindow(τ, ts[j], lb)];
            isempty(v) ? empty : f(v))
        for (ss, col, f, empty) in ((Sum(:x), :x_sum, sum, 0),
            (Min(:x), :x_min, minimum, missing),
            (Opaque(Sum(:x)), :x_sum, sum, 0))
            df = frame(data |> summarizewindows(clk, Month(2), ss), c)
            @test df.time == τs
            @test isequal(df[!, col], [oracle(τ, Month(2), f, empty) for τ in τs])
        end
        # keyed, with a declared key set: every key at every tick
        df = frame(
            data |> summarizewindows(clk, Month(1), Count(); key = :k,
                keyset = ["a", "b"]), c)
        @test df.count == [
            count(j -> ks[j] == k && ts[j] < τ &&
                       inwindow(τ, ts[j], Month(1)), 1:n)
            for τ in τs for k in ("a", "b")
        ]
    end

    @testset "applymodels tolerance" begin
        # a model fit over January, applied to March: reachable with a
        # two-month tolerance, stale with one
        d = readtable(DataFrame(time = ts, x = Float64.(xs), y = 2 .* xs .+ 1.0))
        jan = Context(dt(1, 1), dt(2, 1))
        fits = d |> summarize(FitModel(ToyOLS(), :x, :y))   # emitted at Feb 1
        models = readtable(frame(fits, jan))
        mar = Context(dt(3, 1), dt(4, 1))
        near = frame(d |> applymodels(models; tolerance = Month(2)), mar)
        @test !any(ismissing, near.prediction)
        @test near.prediction ≈ near.y
        far = frame(d |> applymodels(models; tolerance = Month(1)), mar)
        @test all(ismissing, far.prediction[far.time .> dt(3, 1)])
    end
end
