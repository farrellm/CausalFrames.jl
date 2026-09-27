const CF = CausalFrames

# `s`'s value re-folded from scratch over `rows`, the reference for a window.
function barrefold(s, rows, intypes)
    protos, requested, terms = CF.prototypes(CF.Summarizer[s], Symbol[])
    types = CF.barwindowtypes(terms, intypes)
    states = CF.newstates(protos, types)
    for r in rows
        base = NamedTuple{keys(intypes)}(r)
        CF.updateall!(states, merge(base, map(f -> f(base), terms)))
    end
    return CF.summaryvalues(states, Val(requested))
end

barsame(a, b) =
    isequal(a, b) || (a isa Real && b isa Real && isapprox(a, b; rtol = 1e-9))

# Feed `rows` one at a time, checking every value against the re-fold of the
# last n rows (all missing before n).
function checkbarwindow(s, n, rows, intypes)
    st = CF.barwindow(s, n, intypes)
    for i in eachindex(rows)
        CF.update!(st, rows[i])
        v = CF.value(st)
        if i < n
            all(ismissing, v) || return false
        else
            e = barrefold(s, rows[(i-n+1):i], intypes)
            keys(v) == keys(e) || return false
            all(map(barsame, values(v), values(e))) || return false
        end
    end
    return true
end

# Top level, not a testset closure, so the allocation check measures the state.
function barfeed!(st, xs)
    for x in xs
        CF.update!(st, (; x))
        CF.value(st)
    end
    return nothing
end

# The outer kernel's allocations beyond its output vector.
function barfoldalloc(states, nt, r)
    n = length(nt.time)
    CF.foldrunning!(states, nt, n, r)
    total = @allocated CF.foldrunning!(states, nt, n, r)
    return total - @allocated Vector{CF.valuetype(typeof(states), r)}(undef, n)
end

@testset "barwindow" begin
    MF = Union{Missing,Float64}
    # A deterministic spread with NaN and missing runs, so a window holds
    # one, several, then none again.
    xs = MF[
        i % 17 in (3, 4) ? NaN : i % 13 == 5 ? missing :
                                 1 + 0.1 * sin(7.3 * i) for i in 1:300
    ]
    rows = [(x = x,) for x in xs]

    @testset "equals re-folding the last n rows: $(nameof(typeof(s)))" for s in (
        Mean(:x), Std(:x), Sum(:x), Last(:x), Max(:x), First(:x),
        # monoids that are not groups: the two-stack queue
        Product(:x), MinMax(:x), AsMonoid(Max(:x)), AsMonoid(First(:x)),
        AsMonoid(Last(:x)), AsMonoid(Std(:x)))
        for n in (1, 2, 7, 40)
            @test checkbarwindow(s, n, rows, (x = MF,))
        end
    end

    @testset "tiers" begin
        tiers(s) = CF.tiernames(
            CF.tiering(
                CF.prototypes(CF.Summarizer[s], Symbol[])[1], (x = MF,)),
        )
        # The groups slide; the queue is exercised by the non-group monoids.
        @test all(!=(:tree), tiers(Std(:x)))
        @test tiers(Product(:x)) == (:tree,)
        @test tiers(AsMonoid(Max(:x))) == (:tree,)
    end

    @testset "missing until n rows, then Union{Missing, T}" begin
        st = CF.barwindow(Count(), 3, (x = Int,))
        vs = map(1:4) do x
            CF.update!(st, (; x))
            CF.value(st)
        end
        @test eltype(vs) == @NamedTuple{count::Union{Missing,Int}}
        @test isequal(map(v -> v.count, vs), [missing, missing, 3, 3])
    end

    @testset "row terms" begin
        double = r -> 2r.x
        s = Sum(:d => double)
        @test checkbarwindow(s, 4, [(x = Float64(i),) for i in 1:20],
            (x = Float64,))
        # A term shared by two summarizers, read through a dependent.
        @test checkbarwindow(Mean(:d => double), 3,
            [(x = Float64(i),) for i in 1:20], (x = Float64,))
        # The row is converted to `intypes` before the term sees it.
        st = CF.barwindow(Sum(:d => double), 2, (x = Float64,))
        CF.update!(st, (x = 1,))
        CF.update!(st, (x = 2,))
        @test CF.value(st) == (d_sum = 6.0,)
        # Extra row fields are ignored.
        CF.update!(st, (x = 3, y = "z"))
        @test CF.value(st) == (d_sum = 10.0,)
    end

    @testset "fresh and fresh!" begin
        st = CF.barwindow(Product(:x), 2, (x = Float64,))
        foreach(x -> CF.update!(st, (; x)), (2.0, 3.0, 4.0))
        @test CF.value(st) == (x_product = 12.0,)
        f = CF.fresh(st)
        @test typeof(f) == typeof(st)
        @test all(ismissing, CF.value(f))
        @test CF.value(st) == (x_product = 12.0,)
        @test CF.fresh!(st) === st
        @test all(ismissing, CF.value(st))
        foreach(x -> CF.update!(st, (; x)), (5.0, 6.0))
        @test CF.value(st) == (x_product = 30.0,)
    end

    @testset "widening mid-stream" begin
        for s in (Sum(:x), Product(:x), Mean(:x))
            st = CF.barwindow(s, 3, (x = Int,))
            foreach(x -> CF.update!(st, (; x)), 1:5)
            w = CF.widenstate(st, (x = Float64,))
            CF.update!(w, (x = 0.5,))
            @test barsame(only(CF.value(w)),
                only(barrefold(s, [(x = 4.0,), (x = 5.0,), (x = 0.5,)],
                    (x = Float64,))))
        end
        # A widening that stops a group inverting moves it to the queue.
        st = CF.barwindow(FragileSum(:x), 2, (x = Int,))
        foreach(x -> CF.update!(st, (; x)), 1:3)
        w = CF.widenstate(st, (x = Float64,))
        CF.update!(w, (x = 0.5,))
        @test only(CF.value(w)) == 3.5
        # One that starts inverting moves from the queue to the groups.
        st = CF.barwindow(LateSum(:x), 2, (x = Int,))
        foreach(x -> CF.update!(st, (; x)), 1:3)
        w = CF.widenstate(st, (x = Float64,))
        CF.update!(w, (x = 0.5,))
        @test only(CF.value(w)) == 3.5
    end

    @testset "zero allocations per row" begin
        ys = [sin(0.37 * i) for i in 1:200]
        for s in (Std(:x), Max(:x), Product(:x), AsMonoid(Std(:x)),
            Sum(:d => r -> r.x^2))
            st = CF.barwindow(s, 20, (x = Float64,))
            barfeed!(st, ys)
            @test (@allocated barfeed!(st, ys)) == 0
        end
    end

    @testset "embedded in an outer state" begin
        df = DataFrame(time = 1:6, x = Float64[1, 2, 3, 4, 5, 6])
        out = DataFrame(
            load(Context(0, 10),
                readtable(df) |> addsummarycolumns(LastNMean(:x, 3))),
        )
        @test isequal(out.x_lastnmean, [missing, missing, 2.0, 3.0, 4.0, 5.0])
        @test eltype(out.x_lastnmean) == Union{Missing,Float64}
        # The outer kernel's per-row path allocates nothing either.
        protos, requested = CF.prototypes(
            CF.tosummarizers([LastNMean(:x, 3), Std(:x)]), Symbol[])
        states = CF.newstates(protos, (time = Int, x = Float64))
        nt = (time = collect(1:200), x = [sin(0.37 * i) for i in 1:200])
        @test barfoldalloc(states, nt, Val(requested)) == 0
    end

    @testset "validation" begin
        @test_throws ArgumentError("barwindow count must be positive, got 0") CF.barwindow(
            Sum(:x), 0, (x = Int,))
        @test_throws ArgumentError(
            "barwindow summarizers must be MonoidSummarizers, got Opaque{Sum{:x}}",
        ) CF.barwindow(Opaque(Sum(:x)), 3, (x = Int,))
        @test_throws ArgumentError(
            "barwindow row term :x collides with an input column",
        ) CF.barwindow(Sum(:x => r -> 1), 3, (x = Int,))
        opaque = r -> Base.inferencebarrier(r.x)
        @test_throws ArgumentError(
            "barwindow row term :y must infer a concrete type, got Any",
        ) CF.barwindow(Sum(:y => opaque), 3, (x = Int,))
        # A small Union of concrete types is fine.
        @test CF.barwindow(Sum(:y => r -> r.x > 0 ? r.x : missing), 3,
            (x = Int,)) isa SummarizerState
    end
end
