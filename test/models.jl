# Model fitting and prediction, over toy MLJModelInterface models. The oracles
# fit ordinary least squares directly with `pinv`, the same solve ToyOLS uses,
# so agreement is to round-off.
#
# The toy models live here rather than in fixtures.jl because they need
# MLJModelInterface loaded. They must not call `MLJModelInterface.matrix`:
# without MLJBase the interface runs in its light mode, where the table
# helpers throw, so tables go through Tables.matrix instead.
const MMI = MLJModelInterface
using LinearAlgebra: pinv

tablerows(X) = size(Tables.matrix(X), 1)

# Ordinary least squares with an intercept, as the minimum-norm solution, so a
# window too short or too degenerate to determine the fit (two rows sharing an
# x, say) still fits rather than throwing; the report records the fit's size.
mutable struct ToyOLS <: MMI.Deterministic end
function MMI.fit(::ToyOLS, verbosity::Int, X, y)
    A = hcat(ones(length(y)), Tables.matrix(X))
    return pinv(A) * Float64.(y), nothing, (; n = length(y), p = size(A, 2))
end
function MMI.predict(::ToyOLS, β, Xnew)
    M = Tables.matrix(Xnew)
    return hcat(ones(size(M, 1)), M) * β
end

# Keeps its training table in the fitresult, so a test can see whether the
# rows it was handed were disturbed afterwards.
mutable struct Hoarder <: MMI.Deterministic end
MMI.fit(::Hoarder, verbosity::Int, X, y) = ((X = X, y = y), nothing, nothing)
MMI.predict(::Hoarder, fr, Xnew) = fill(float(sum(fr.y)), tablerows(Xnew))

# A probabilistic model whose "distribution" is a NamedTuple, with its mean as
# the predict_mean.
mutable struct ToyMean <: MMI.Probabilistic end
MMI.fit(::ToyMean, verbosity::Int, X, y) =
    (sum(y) / length(y), nothing, (; n = length(y)))
MMI.predict(::ToyMean, μ, Xnew) = fill((mean = μ, spread = 1.0), tablerows(Xnew))
MMI.predict_mean(::ToyMean, μ, Xnew) = fill(μ, tablerows(Xnew))

# A fitresult standing in for a foreign resource: `save` swaps the handle for a
# plain number and `restore` rebuilds it, so a round trip skipping either would
# hand predict a Float64 where it needs a Handle. The counters prove both ran.
struct Handle
    value::Float64
end
const POINTERSAVES = Ref(0)
const POINTERRESTORES = Ref(0)
mutable struct PointerModel <: MMI.Deterministic end
MMI.fit(::PointerModel, verbosity::Int, X, y) =
    (Handle(sum(y) / length(y)), nothing, nothing)
MMI.predict(::PointerModel, h::Handle, Xnew) = fill(h.value, tablerows(Xnew))
MMI.save(::PointerModel, h::Handle; kwargs...) = (POINTERSAVES[] += 1; h.value)
MMI.restore(::PointerModel, v::Float64) = (POINTERRESTORES[] += 1; Handle(v))

olsfit(x::AbstractVector, y) = pinv(hcat(ones(length(x)), x)) * Float64.(y)

onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])
# A source clipping its chunks to the context, as real sources do, so the
# widened contexts of the window operators decide which early rows are seen.
clippedsource(chunks) = CausalPipeline(
    ctx -> filter(c -> nrow(c) > 0,
        [c[(ctx.start .<= c.time) .& (c.time .< ctx.stop), :] for c in chunks]))

# The prediction oracle for addpredictions: each row at t is predicted by OLS
# fit on the rows (of its key) in [τ - L, τ), τ the latest tick <= t; missing
# before the first tick and wherever that window is empty.
function predictionoracle(df::DataFrame, rows, ticks, L; key = nothing)
    out = Vector{Union{Missing,Float64}}(missing, length(rows))
    for (o, i) in enumerate(rows)
        t = df.time[i]
        j = findlast(<=(t), ticks)
        j === nothing && continue
        τ = ticks[j]
        sel = (τ - L .<= df.time) .& (df.time .< τ)
        key === nothing || (sel .&= isequal.(df[!, key], df[i, key]))
        any(sel) || continue
        β = olsfit(df.x[sel], df.y[sel])
        out[o] = β[1] + β[2] * df.x[i]
    end
    return out
end

predsame(x, y) =
    ismissing(x) ? ismissing(y) : !ismissing(y) && isapprox(x, y; atol = 1e-8)

@testset "FitModel" begin
    n = 60
    xs = Float64.(lcgsequence(21, n, 11))
    zs = Float64.(lcgsequence(22, n, 7))
    ys = 2 .+ 3 .* xs .- zs .+ 0.1 .* Float64.(lcgsequence(23, n, 5))
    p = onechunk(time = collect(1:n), x = xs, z = zs, y = ys)
    ctx = Context(0, 100)

    @testset "summarize fits once; coefficients match LinearRegression" begin
        df = DataFrame(load(ctx, p |> summarize(FitModel(ToyOLS(), [:x, :z], :y))))
        @test names(df) == ["time", "model"]
        fm = only(df.model)
        @test fm isa FittedModel{(:x, :z),ToyOLS}
        lr = DataFrame(load(ctx, p |> summarize(LinearRegression([:x, :z], :y))))
        @test fm.fitresult ≈ [only(lr.intercept_beta), only(lr.x_beta),
            only(lr.z_beta)]
        @test fm.report == (; n = n, p = 3)
        @test fm.model isa ToyOLS

        # a lone predictor name, and the output name
        df = DataFrame(load(ctx,
            p |> summarize(FitModel(ToyOLS(), :x, :y; name = :fit))))
        @test names(df) == ["time", "fit"]
        @test only(df.fit).fitresult ≈ olsfit(xs, ys)
    end

    @testset "one fit per interval, per key" begin
        ks = map(v -> ("a", "b")[v+1], lcgsequence(24, n, 2))
        q = onechunk(time = collect(1:n), k = ks, x = xs, y = ys)
        df = DataFrame(
            load(ctx,
                q |> intervalize(clock(20), FitModel(ToyOLS(), :x, :y); key = :k)),
        )
        for r in eachrow(df)
            sel = (r.time - 20 .<= (1:n) .< r.time) .& (ks .== r.k)
            @test r.model.fitresult ≈ olsfit(xs[sel], ys[sel])
        end
        # one row per (complete interval, key present in it); row 60 opens a
        # fourth interval [60, 80) with a single key
        @test nrow(df) ==
              sum(length(unique(ks[b .<= (1:n) .< b+20])) for b in 0:20:60)
    end

    @testset "empty input, widening" begin
        df = DataFrame(load(ctx,
            emptyframe() |> summarize(FitModel(ToyOLS(), :x, :y))))
        @test isequal(df.model, [missing])

        # x arrives Int in the first chunk and Float64 in the second
        ix = lcgsequence(25, n, 9)
        mixed = CausalPipeline(
            ctx -> [
                DataFrame(time = 1:30, x = ix[1:30], y = ys[1:30]),
                DataFrame(time = 31:n, x = Float64.(ix[31:n]), y = ys[31:n])],
        )
        df = DataFrame(load(ctx, mixed |> summarize(FitModel(ToyOLS(), :x, :y))))
        @test only(df.model).fitresult ≈ olsfit(Float64.(ix), ys)
    end

    @testset "a model keeping its training table is not disturbed by refits" begin
        # The re-fold path reuses the state's buffers from window to window;
        # FitModel hands the model copies, so a fitresult holding its X keeps
        # exactly its own window's rows.
        df = DataFrame(
            load(Context(10, 60),
                p |> summarizewindows(clock(10), 15, FitModel(Hoarder(), :x, :y))),
        )
        for r in eachrow(df)
            sel = r.time - 15 .<= (1:n) .< r.time
            @test r.model.fitresult.X.x == xs[sel]
            @test r.model.fitresult.y == ys[sel]
        end
    end

    @testset "construction errors" begin
        @test_throws ArgumentError FitModel(42, [:x], :y)
        @test_throws ArgumentError FitModel(ToyOLS(), Symbol[], :y)
        @test_throws ArgumentError FitModel(ToyOLS(), [:x, :x], :y)
        @test_throws ArgumentError FitModel(ToyOLS(), [:x, :y], :y)
    end

    @testset "without MLJModelInterface" begin
        # the model can't exist without the package, so any value is checked;
        # the message must say what to load
        @test insubprocess("""
        using CausalFrames
        try
            FitModel(nothing, [:x], :y)
            error("expected an ArgumentError")
        catch e
            e isa ArgumentError || rethrow()
            occursin("MLJModelInterface", e.msg) || rethrow()
        end
        print("ok")
        """) == "ok"
    end
end

@testset "applymodels" begin
    n = 80
    ts = collect(1:n)
    xs = Float64.(lcgsequence(31, n, 13))
    ys = 1 .+ 2 .* xs .+ 0.5 .* Float64.(lcgsequence(32, n, 3))
    ks = map(v -> ("a", "b")[v+1], lcgsequence(33, n, 2))
    data = onechunk(time = ts, k = ks, x = xs, y = ys)
    ctx = Context(0, 100)
    fits = data |> summarizewindows(clock(20), 20, FitModel(ToyOLS(), :x, :y))

    @testset "each row takes the latest model at or before it" begin
        df = DataFrame(load(ctx, data |> applymodels(fits)))
        @test names(df) == ["time", "k", "x", "y", "prediction"]
        @test eltype(df.prediction) == Union{Missing,Float64}
        want = map(1:n) do i
            τ = 20 * (ts[i] ÷ 20)
            τ == 0 && return missing          # the tick-0 window is empty
            β = olsfit(xs[τ-20 .<= ts .< τ], ys[τ-20 .<= ts .< τ])
            β[1] + β[2] * xs[i]
        end
        @test all(map(predsame, df.prediction, want))

        # strict: a row exactly at a tick takes the previous tick's model
        st = DataFrame(load(ctx, data |> applymodels(fits; strict = true)))
        at40 = findfirst(==(40), ts)
        β20 = olsfit(xs[0 .<= ts .< 20], ys[0 .<= ts .< 20])
        @test st.prediction[at40] ≈ β20[1] + β20[2] * xs[at40]
        @test df.prediction[at40] ≉ st.prediction[at40]

        # the uncurried form
        @test isequal(DataFrame(load(ctx, applymodels(data, fits))), df)
    end

    @testset "per-key models" begin
        kfits =
            data |> summarizewindows(clock(20), 20, FitModel(ToyOLS(), :x, :y);
                key = :k)
        df = DataFrame(load(ctx, data |> applymodels(kfits; key = :k)))
        @test all(
            map(predsame, df.prediction,
                predictionoracle(DataFrame(time = ts, k = ks, x = xs, y = ys), 1:n,
                    0:20:99, 20; key = :k)),
        )
    end

    @testset "models from an earlier window, through a jls file and tolerance" begin
        dir = mktempdir()
        path = joinpath(dir, "models.jls")
        # a source that clips to its window, since this test runs over two
        # different ones: the fit must see only [0, 40), the predictions only
        # [50, 80)
        clipped = clippedsource([DataFrame(time = ts, k = ks, x = xs, y = ys)])
        # one model, fit on [0, 40) and emitted at 40
        scan(
            Context(0, 40),
            clipped |> summarize(FitModel(ToyOLS(), :x, :y)) |>
            writejls(path),
        )
        later = Context(50, 80)
        # without tolerance the model row at 40 is outside the window
        df = DataFrame(load(later, clipped |> applymodels(readjls(path))))
        @test all(ismissing, df.prediction)
        # with it, rows up to 40 + 25 use the model and later rows do not
        df = DataFrame(load(later, clipped |> applymodels(readjls(path);
            tolerance = 25)))
        β = olsfit(xs[ts .< 40], ys[ts .< 40])
        for r in eachrow(df)
            r.time <= 65 ? (@test r.prediction ≈ β[1] + β[2] * r.x) :
            (@test ismissing(r.prediction))
        end
    end

    @testset "missing models, an empty models stream, operations" begin
        # the tick-0 window is empty, so rows before 20 match a missing model
        df = DataFrame(load(ctx, data |> applymodels(fits)))
        @test all(ismissing, df.prediction[ts .< 20])

        df = DataFrame(load(ctx, data |> applymodels(emptyframe())))
        @test eltype(df.prediction) == Missing
        @test nrow(df) == n

        # a probabilistic model: :predict gives its "distributions",
        # :predict_mean their means
        pfits = data |> summarizewindows(clock(20), 20, FitModel(ToyMean(), :x, :y))
        dist = DataFrame(load(ctx, data |> applymodels(pfits)))
        mean = DataFrame(
            load(ctx, data |> applymodels(pfits;
                    operation = :predict_mean, name = :yhat)),
        )
        @test names(mean)[end] == "yhat"
        i = findfirst(>=(30), ts)
        @test dist.prediction[i] == (mean = mean.yhat[i], spread = 1.0)
        # the row at 30 uses the tick-20 model, fit on [0, 20)
        @test mean.yhat[i] ≈ sum(ys[ts .< 20]) / count(ts .< 20)
    end

    @testset "errors" begin
        @test_throws ArgumentError applymodels(fits; operation = :transform)
        @test_throws ArgumentError applymodels(fits; key = :k, name = :k)
        @test_throws ArgumentError applymodels(fits; column = :time)
        @test_throws ArgumentError applymodels(fits; key = :time)
        # at run time: the output name collides with an input column
        @test_throws ArgumentError load(ctx, data |> applymodels(fits; name = :x))
        # the models pipeline has no such column
        @test_throws ArgumentError load(ctx,
            data |> applymodels(fits; column = :nope))
        # a predictor the input lacks
        @test_throws ArgumentError load(ctx,
            onechunk(time = ts, z = xs) |> applymodels(fits))
        # a cell that is not a model
        notmodels = onechunk(time = [10], model = [1])
        @test_throws ArgumentError load(ctx, data |> applymodels(notmodels))
        # models keyed on a column the models pipeline lacks
        @test_throws ArgumentError load(ctx, data |> applymodels(fits; key = :k))
    end
end

@testset "addpredictions" begin
    n = 300
    ts = cumsum(lcgsequence(41, n, 3))     # 0-2 steps, with ties
    xs = Float64.(lcgsequence(42, n, 17))
    ys = 1 .+ 2 .* xs .+ 0.25 .* Float64.(lcgsequence(43, n, 7))
    ks = map(v -> ("a", "b", "c")[v+1], lcgsequence(44, n, 3))
    whole = DataFrame(time = ts, k = ks, x = xs, y = ys)
    ranges = [1:100, 101:220, 221:n]
    p = clippedsource([whole[r, :] for r in ranges])
    ctx = Context(40, 250)
    inwindow = findall(t -> 40 <= t < 250, ts)
    ticks = 40:15:249

    for key in (nothing, :k), L in (10, 40)
        df = DataFrame(load(ctx,
            p |> addpredictions(clock(15), L, ToyOLS(), :x, :y; key)))
        @test names(df) == ["time", "k", "x", "y", "prediction"]
        want = predictionoracle(whole, inwindow, ticks, L; key)
        @test all(map(predsame, df.prediction, want))
        # the first tick's window reaches back before `start`, so its rows are
        # predicted (the oracle agreeing is the check; this is the witness)
        @test any(!ismissing, df.prediction[df.time .< 55])
    end

    # a key whose data stops is predicted missing once its window empties
    q = onechunk(time = [1, 2, 3, 30, 31], k = ["a", "a", "b", "a", "b"],
        x = [1.0, 2.0, 1.0, 3.0, 2.0], y = [3.0, 5.0, 2.0, 7.0, 4.0])
    df = DataFrame(
        load(Context(0, 40),
            q |> addpredictions(clock(10), 10, ToyOLS(), :x, :y; key = :k)),
    )
    @test all(ismissing, df.prediction)   # ticks 10, 20, 30: windows before 10 only

    t = addpredictions(clock(15), 40, ToyOLS(), :x, :y; key = :k)
    loaded = DataFrame(load(ctx, p |> t))
    streamed = reduce(vcat, DataFrame.(stream(ctx, p |> t)))
    @test isequal(streamed, loaded)

    # the uncurried form
    @test isequal(
        DataFrame(load(ctx,
            addpredictions(p, clock(15), 40, ToyOLS(), :x, :y; key = :k))), loaded)
end

@testset "modelreports" begin
    n = 50
    ks = map(v -> ("a", "b")[v+1], lcgsequence(51, n, 2))
    xs = Float64.(lcgsequence(52, n, 9))
    data = onechunk(time = collect(1:n), k = ks, x = xs, y = 2 .* xs)
    ctx = Context(0, 60)

    fits = data |> summarizewindows(clock(10), 10, FitModel(ToyOLS(), :x, :y))
    df = DataFrame(load(ctx, fits |> modelreports()))
    @test names(df) == ["time", "report"]
    @test df.time == 0:10:50
    @test ismissing(df.report[1])      # the tick-0 window is empty
    @test [r.n for r in df.report[2:end]] == [9, 10, 10, 10, 10]
    @test eltype(df.report) == Union{Missing,@NamedTuple{n::Int,p::Int}}

    # keyed, with the key passing through and a renamed column
    kfits =
        data |> summarizewindows(clock(10), 10,
            FitModel(ToyOLS(), :x, :y; name = :m); key = :k)
    df = DataFrame(load(ctx, kfits |> modelreports(; column = :m, name = :diag)))
    @test names(df) == ["time", "k", "diag"]
    for r in eachrow(df)
        sel = (r.time - 10 .<= (1:n) .< r.time) .& (ks .== r.k)
        ismissing(r.diag) ? (@test !any(sel)) : (@test r.diag.n == count(sel))
    end

    # the uncurried form, and streaming
    @test isequal(DataFrame(load(ctx, modelreports(fits))),
        DataFrame(load(ctx, fits |> modelreports())))
    @test isequal(reduce(vcat, DataFrame.(stream(ctx, fits |> modelreports()))),
        DataFrame(load(ctx, fits |> modelreports())))

    @test_throws ArgumentError modelreports(; column = :time)
    @test_throws ArgumentError modelreports(; name = :time)
    @test_throws ArgumentError load(ctx, fits |> modelreports(; column = :nope))
    @test_throws ArgumentError load(ctx, kfits |> modelreports(; column = :m,
        name = :k))
    @test_throws ArgumentError load(ctx,
        onechunk(time = [1], model = [1]) |> modelreports())
end

@testset "fitted models round-trip through jls" begin
    dir = mktempdir()
    n = 40
    xs = Float64.(lcgsequence(61, n, 9))
    data = onechunk(time = collect(1:n), x = xs, y = 3 .* xs)
    ctx = Context(0, 50)

    # an ordinary model: the whole table, reports included, comes back equal
    fits = data |> summarizewindows(clock(10), 10, FitModel(ToyOLS(), :x, :y))
    path = joinpath(dir, "ols.jls")
    scan(ctx, fits |> writejls(path))
    back = DataFrame(load(ctx, readjls(path)))
    orig = DataFrame(load(ctx, fits))
    @test isequal(map(m -> ismissing(m) ? m : m.fitresult, back.model),
        map(m -> ismissing(m) ? m : m.fitresult, orig.model))
    @test isequal(DataFrame(load(ctx, readjls(path) |> modelreports())),
        DataFrame(load(ctx, fits |> modelreports())))
    @test isequal(DataFrame(load(ctx, data |> applymodels(readjls(path)))),
        DataFrame(load(ctx, data |> applymodels(fits))))

    # a model whose fitresult needs MLJ's save/restore: restore rebuilds the
    # live handle, without which prediction would fail outright
    POINTERSAVES[] = 0
    POINTERRESTORES[] = 0
    pfits = data |> summarizewindows(clock(10), 10, FitModel(PointerModel(), :x, :y))
    ppath = joinpath(dir, "pointer.jls")
    scan(ctx, pfits |> writejls(ppath))
    @test POINTERSAVES[] == 4                   # ticks 10..40; tick 0 is empty
    df = DataFrame(load(ctx, data |> applymodels(readjls(ppath))))
    @test POINTERRESTORES[] == 4
    @test isequal(df.prediction,
        DataFrame(load(ctx, data |> applymodels(pfits))).prediction)
end
