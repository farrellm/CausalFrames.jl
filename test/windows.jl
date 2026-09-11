# A group summarizer whose state reports itself non-invertible once its column
# widens to Float64. Every built-in accumulator stays invertible under
# widening, so this is the only way to drive summarizewindows' running -> tree
# demotion. The output name rides in a type parameter, as the
# interface requires for `value` to infer.
struct FragileSum{C} <: GroupSummarizer end
FragileSum(column::Symbol) = FragileSum{column}()
mutable struct FragileSumState{C,N,T} <: SummarizerState
    total::T
end
CausalFrames.emptyvalue(::FragileSum{C}) where {C} =
    NamedTuple{(Symbol(C, :_fragile),)}((0,))
CausalFrames.fresh(::FragileSum{C}, intypes::NamedTuple) where {C} =
    FragileSumState{C,Symbol(C, :_fragile),intypes[C]}(zero(intypes[C]))
CausalFrames.fresh(::FragileSumState{C,N,T}) where {C,N,T} =
    FragileSumState{C,N,T}(zero(T))
CausalFrames.update!(st::FragileSumState{C}, row) where {C} =
    (st.total += getproperty(row, C); nothing)
CausalFrames.downdate!(st::FragileSumState{C}, row) where {C} =
    (st.total -= getproperty(row, C); nothing)
CausalFrames.combine!(dest::FragileSumState, a, b) =
    (dest.total = a.total + b.total; nothing)
CausalFrames.value(st::FragileSumState{C,N,T}) where {C,N,T} =
    NamedTuple{(N,),Tuple{T}}((st.total,))
CausalFrames.widenstate(st::FragileSumState{C,N}, intypes::NamedTuple) where {C,N} =
    FragileSumState{C,N,intypes[C]}(convert(intypes[C], st.total))
CausalFrames.isinvertible(::FragileSumState{C,N,Float64}) where {C,N} = false

# The brute-force oracle: at each tick τ, `summarize` the rows in [τ - L, τ)
# alone, then apply the keyed vanish rule — an empty row for every key present
# at the previous tick and absent now — and sort each tick's rows by key.
function windowsoracle(df::DataFrame, ticks, L, ss; key = nothing)
    keycols = CausalFrames.tokeycolumns(key)
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers(ss), keycols)
    empty = CausalFrames.emptyvalues(protos, Val(requested))
    parts = DataFrame[]
    prev = NamedTuple[]
    for τ in ticks
        w = df[(τ-L .<= df.time) .& (df.time .< τ), :]
        src = CausalPipeline(ctx -> nrow(w) > 0 ? [copy(w)] : DataFrame[])
        got = DataFrame(load(Context(τ - L, τ), src |> summarize(ss; key)))
        if !isempty(keycols)
            present = [
                NamedTuple{Tuple(keycols)}(Tuple(r[k] for k in keycols))
                for r in eachrow(got)
            ]
            vanished = [k for k in prev if !any(isequal(k), present)]
            if !isempty(vanished)
                vdf = DataFrame([merge((; time = τ), k, empty) for k in vanished])
                got = nrow(got) == 0 ? vdf : vcat(got, vdf; cols = :setequal)
                sort!(got, keycols)
            end
            prev = present
        end
        nrow(got) > 0 && push!(parts, got)
    end
    return isempty(parts) ? DataFrame() : reduce(vcat, parts)
end

# Column-wise agreement, as in the rolling tests: exact for everything but
# floats, which may carry the running path's compensated round-off (and NaN,
# which isapprox rejects). A bare `atol` would zero isapprox's relative
# tolerance, so the default is kept.
windowsame(x, y) =
    ismissing(x) ? ismissing(y) :
    !ismissing(y) && (isapprox(x, y) || (isnan(x) && isnan(y)))
function windowsagree(a::DataFrame, b::DataFrame)
    @test nrow(a) == nrow(b)
    nrow(a) == nrow(b) == 0 && return
    @test names(a) == names(b)
    (nrow(a) == nrow(b) && names(a) == names(b)) || return
    for n in names(a)
        x, y = a[!, n], b[!, n]
        if nonmissingtype(eltype(x)) <: AbstractFloat ||
           nonmissingtype(eltype(y)) <: AbstractFloat
            @test all(map(windowsame, x, y))
        else
            @test isequal(x, y)
        end
    end
end

@testset "summarizewindows" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])
    # A source that clips its chunks to the context, as real sources do, so
    # the widened input context is what decides which early rows are seen.
    clipped(chunks) = CausalPipeline(
        ctx -> filter(c -> nrow(c) > 0,
            [c[(ctx.start .<= c.time) .& (c.time .< ctx.stop), :] for c in chunks]))

    @testset "keyless grid over trailing windows" begin
        p = onechunk(time = [2, 7, 7, 12], x = [10, 20, 30, 40])
        run(L) = DataFrame(
            load(Context(0, 20),
                p |> summarizewindows(clock(5), L, [Count(), Sum(:x), Mean(:x)])),
        )
        df = run(5)
        @test names(df) == ["time", "count", "x_sum", "x_mean"]
        @test df.time == [0, 5, 10, 15]
        @test df.count == [0, 1, 2, 1]
        @test df.x_sum == [0, 10, 50, 40]
        @test isequal(df.x_mean, [missing, 10.0, 25.0, 40.0])
        @test eltype(df.x_mean) == Union{Missing,Float64}

        # overlapping windows: a look-back longer than the tick spacing
        df = run(10)
        @test df.count == [0, 1, 3, 3]
        @test df.x_sum == [0, 10, 60, 90]

        # a single summarizer (not a collection), and the uncurried form
        @test DataFrame(load(Context(0, 20),
            summarizewindows(p, clock(5), 5, Count()))).count == [0, 1, 2, 1]
    end

    @testset "membership is half-open at the tick" begin
        # a row exactly at tick 5 belongs to the window closing at 10
        p = onechunk(time = [5], x = [99])
        df = DataFrame(load(Context(0, 15),
            p |> summarizewindows(clock(5), 5, Sum(:x))))
        @test df.time == [0, 5, 10]
        @test df.x_sum == [0, 0, 99]
    end

    @testset "rows past the clock's last tick are released" begin
        # the clock ends at 10 while the data runs on: the second chunk falls in
        # no window and is dropped, in the running and the tree modes alike
        p = CausalPipeline(
            ctx -> [DataFrame(time = [2, 7, 12], x = [1, 2, 3]),
                DataFrame(time = [20, 30], x = [4, 5])],
        )
        ticks = CausalPipeline(ctx -> [DataFrame(time = [5, 10])])
        for ss in ([Count(), Sum(:x)], Min(:x))
            df = DataFrame(load(Context(0, 40),
                p |> summarizewindows(ticks, 5, ss)))
            @test df.time == [5, 10]
            @test df[!, end] == [1, 2]
        end
        none = CausalPipeline(ctx -> DataFrame[])
        @test nrow(
            DataFrame(load(Context(0, 40),
                p |> summarizewindows(none, 5, Sum(:x)))),
        ) == 0
    end

    @testset "the first tick sees a full window from before start" begin
        p = clipped([DataFrame(time = [3, 7, 8, 12], x = [1, 2, 3, 4])])
        df = DataFrame(load(Context(10, 20),
            p |> summarizewindows(clock(5), 5, Sum(:x))))
        @test df.time == [10, 15]
        @test df.x_sum == [5, 4]     # rows 7 and 8 (row 3 is too old), then 12
    end

    @testset "keyed windows are sparse, with one vanish row" begin
        p = onechunk(time = [1, 1, 12], k = ["a", "b", "b"], x = [1, 2, 3])
        df = DataFrame(
            load(Context(5, 25),
                p |> summarizewindows(clock(5), 5, [Count(), Min(:x)]; key = :k)),
        )
        @test names(df) == ["time", "k", "count", "x_min"]
        # 5: a, b present; 10: both vanish; 15: b back; 20: b vanishes; the
        # key a, once vanished, is never emitted again
        @test df.time == [5, 5, 10, 10, 15, 20]
        @test df.k == ["a", "b", "a", "b", "b", "b"]
        @test df.count == [1, 1, 0, 0, 1, 0]
        @test isequal(df.x_min, [1, 2, missing, missing, 3, missing])
        @test eltype(df.x_min) == Union{Missing,Int}

        # the vanish row is interleaved in key order with present keys
        p = onechunk(time = [1, 1, 6, 6], k = ["b", "c", "a", "c"], x = [1, 2, 3, 4])
        df = DataFrame(
            load(Context(5, 15),
                p |> summarizewindows(clock(5), 5, Count(); key = :k)),
        )
        @test df.time == [5, 5, 10, 10, 10]
        @test df.k == ["b", "c", "a", "b", "c"]
        @test df.count == [1, 1, 1, 0, 1]
    end

    @testset "a look-back equal to the spacing reproduces intervalize" begin
        times = cumsum(lcgsequence(11, 200, 3))
        xs = lcgsequence(12, 200, 9)
        p = clipped([DataFrame(time = times, x = xs)])
        ss = [Count(), Sum(:x), Mean(:x), Min(:x)]
        ctx = Context(10, 300)
        w = DataFrame(load(ctx, p |> summarizewindows(clock(10), 10, ss)))
        i = DataFrame(load(ctx, p |> intervalize(clock(10), ss)))
        windowsagree(w[w.time .> 10, :], i)
    end

    # A pseudo-random three-chunk stream with tied times and three keys, starting
    # before the context so the first ticks' windows reach back past `start`.
    nrows = 300
    times = cumsum(lcgsequence(1, nrows, 3))           # 0-2 steps: many ties
    xs = map(v -> v - 3, lcgsequence(2, nrows, 7))
    ys = map(v -> v - 2, lcgsequence(3, nrows, 5))
    ks = map(v -> ("a", "b", "c")[v+1], lcgsequence(4, nrows, 3))
    ranges = [1:100, 101:220, 221:300]
    frame(x) = DataFrame(time = times, k = ks, x = x, y = ys)
    mkdata(x) = clipped([frame(x)[r, :] for r in ranges])
    intx, floatx = xs, Float64.(xs) ./ 4
    ctx = Context(20, 250)
    ticks = 20:5:245

    groupset = [Count(), Sum(:x), Mean(:x), Variance(:x), Correlation(:x, :y),
        LinearRegression(:x, :y; name = :m1)]
    monoidset = [Min(:x), Max(:x), First(:x), Last(:x), CountDistinct(:x)]
    plainset = [Sum(:x), TestVar(:x)]
    windowed(p, L, ss; kwargs...) = DataFrame(load(ctx,
        p |> summarizewindows(clock(5), L, ss; kwargs...)))

    @testset "differential against the brute-force oracle" begin
        for x in (intx, floatx), ss in (groupset, monoidset, plainset),
            L in (3, 11), key in (nothing, :k)

            windowsagree(windowed(mkdata(x), L, ss; key),
                windowsoracle(frame(x), ticks, L, ss; key))
        end
    end

    @testset "running agrees with re-fold" begin
        for x in (intx, floatx), key in (nothing, :k)
            windowsagree(windowed(mkdata(x), 11, groupset; key),
                windowed(mkdata(x), 11, map(Opaque, groupset); key))
        end
    end

    @testset "tree agrees with re-fold" begin
        # L = 40 spans eight ticks, so trees grow, rebuild and slide; the mixed
        # set is all monoid (Sum is a group, hence a monoid) and takes the tree
        mixedset = [Sum(:x), Min(:x), Last(:x)]
        for x in (intx, floatx), ss in (monoidset, mixedset), L in (3, 11, 40),
            key in (nothing, :k)

            windowsagree(windowed(mkdata(x), L, ss; key),
                windowed(mkdata(x), L, map(Opaque, ss); key))
        end
    end

    @testset "widening, and demotion from running to tree" begin
        # x arrives Int in the first chunk and Float64 afterwards
        mixed = CausalPipeline(
            ctx -> [
                DataFrame(time = times[r], k = ks[r],
                    x = r == 1:100 ? xs[r] : Float64.(xs[r]), y = ys[r])
                for r in ranges
            ])
        whole = frame(Float64.(xs))
        # running stays running, running demotes to tree, tree widens within
        # tree, re-fold widens within re-fold
        for ss in ([Sum(:x), Mean(:x)], [FragileSum(:x), Count()],
                [Min(:x), Last(:x)], [Sum(:x), TestVar(:x)]),
            key in (nothing, :k)

            windowsagree(windowed(mixed, 11, ss; key),
                windowsoracle(whole, ticks, 11, ss; key))
        end
    end

    @testset "streaming agrees with loading" begin
        for ss in (groupset, monoidset), key in (nothing, :k)
            t = summarizewindows(clock(5), 11, ss; key)
            loaded = DataFrame(load(ctx, mkdata(intx) |> t))
            streamed = reduce(vcat, DataFrame.(stream(ctx, mkdata(intx) |> t)))
            @test isequal(streamed, loaded)
        end
    end

    @testset "Dates" begin
        t0 = DateTime(2026, 1, 1)
        p = onechunk(time = t0 .+ Minute.([5, 25, 35, 80]), x = [1, 2, 3, 4])
        df = DataFrame(
            load(Context(t0, t0 + Hour(2)),
                p |> summarizewindows(clock(Minute(30)), Minute(45), Sum(:x))),
        )
        @test df.time == t0 .+ Minute.([0, 30, 60, 90])
        @test df.x_sum == [0, 3, 5, 4]
    end

    @testset "empty data and empty clock" begin
        # no data: a keyless grid of empty rows, typed from the configs
        df = DataFrame(
            load(Context(0, 15),
                emptyframe() |> summarizewindows(clock(5), 5, [Count(), Mean(:x)])),
        )
        @test df.time == [0, 5, 10]
        @test df.count == [0, 0, 0]
        @test all(ismissing, df.x_mean)
        # keyed: nothing at all
        @test nrow(
            DataFrame(
                load(Context(0, 15),
                    emptyframe() |> summarizewindows(clock(5), 5, Count(); key = :k)),
            ),
        ) == 0
        # no ticks: nothing, whatever the data
        p = onechunk(time = [1, 2], x = [1, 2])
        @test nrow(
            DataFrame(
                load(Context(0, 15),
                    p |> summarizewindows(emptyframe(), 5, Count())),
            ),
        ) == 0
    end

    @testset "errors" begin
        p = onechunk(time = [1], k = [1], x = [1])
        @test_throws ArgumentError summarizewindows(clock(5), 5, Count();
            key = :time)
        @test_throws ArgumentError summarizewindows(clock(5), 5, Count();
            key = [:k, :k])
        # a summary output named like a key column
        @test_throws ArgumentError summarizewindows(clock(5), 5, Count();
            key = :count)
        @test_throws ArgumentError load(Context(0, 10),
            p |> summarizewindows(clock(5), -1, Count()))
        @test_throws ArgumentError load(Context(0, 10),
            p |> summarizewindows(clock(5), 5, Count(); key = :nope))
    end
end
