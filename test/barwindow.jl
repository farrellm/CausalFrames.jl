const CF = CausalFrames

# `s`'s value re-folded from scratch over `rows`, the reference for a window.
function barrefold(s, rows, intypes)
    protos, requested, terms = CF.prototypes(CF.Summarizer[s], Symbol[])
    types = last(CF.barwindowtypes(terms, intypes, protos))
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
    CF.foldrunning!(states, states, nt, n, Val(()), r)
    total = @allocated CF.foldrunning!(states, states, nt, n, Val(()), r)
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
        @test all(!=(:monoid), tiers(Std(:x)))
        @test tiers(Product(:x)) == (:monoid,)
        @test tiers(AsMonoid(Max(:x))) == (:monoid,)
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
        # The term reads the whole row, but only the columns the window's
        # summarizers read are stored, with `:time` and the terms.
        @test fieldnames(eltype(st.ring.rows)) == (:d,)
        wide = (time = Int, x = Float64, y = Float64, z = String)
        tst = CF.barwindow(Covariance(:y, :d => double), 2, wide)
        @test fieldnames(eltype(tst.ring.rows)) == (:time, :y, :d)
        foreach(i -> CF.update!(tst, (time = i, x = i, y = -i, z = "z")), 1:3)
        @test CF.value(tst) == (y_d_covariance = -1.0,)
        wider = merge(wide, (y = Union{Missing,Float64},))
        @test CF.value(CF.widenstate(tst, wider)) == (y_d_covariance = -1.0,)
        # all of them when a summarizer may read any column
        rst = CF.barwindow(RangeTotal(), 2, (high = Float64, low = Float64, z = String))
        @test fieldnames(eltype(rst.ring.rows)) == (:high, :low, :z, :range)
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
        # groups only: no queue to reset
        st = CF.barwindow(Mean(:x), 2, (x = Float64,))
        foreach(x -> CF.update!(st, (; x)), (2.0, 3.0, 4.0))
        @test CF.fresh!(st) === st
        @test all(ismissing, CF.value(st))
        foreach(x -> CF.update!(st, (; x)), (5.0, 6.0))
        @test CF.value(st) == (x_mean = 5.5,)
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

    @testset "barinputcolumns" begin
        @test CF.barinputcolumns(Sum(:x)) == (:x,)
        # dependencies expanded, each column once
        @test CF.barinputcolumns(Std(:x)) == (:x,)
        @test CF.barinputcolumns(Covariance(:x, :y)) == (:x, :y)
        @test CF.barinputcolumns(Count()) == ()
        # a summarizer that may read any column, and an inner row term,
        # keep every column
        @test CF.barinputcolumns(Opaque(Sum(:x))) === nothing
        @test CF.barinputcolumns(Mean(:d => r -> 2r.x)) === nothing
        @test CF.barinputcolumns(RangeTotal()) === nothing
        @test CF.inputcolumns(LastNMean(:x, 3)) == (:x,)
        # a hoisted term is a stored column
        owner = LastN(Sum(:x), 3)
        @test CF.barinputcolumns(Covariance(:x, :d => r -> 2r.x);
            hoistedby = owner) == (:d, :x, Symbol("#barterm#lastn#d"))
        @test CF.barinputcolumns(RangeTotal(); hoistedby = owner) === nothing
    end

    @testset "hoisted row terms" begin
        double = r -> 2r.x
        collides = ArgumentError("barwindow row term :d collides with an input column")
        notfound = ArgumentError(
            "barwindow hoisted row term :d not found in the input; " *
            "return `barterms(s; hoistedby)` from the outer summarizer's `rowterms`",
        )
        owner = LastN(Sum(:x), 3)
        @test CF.barterms(Sum(:x); hoistedby = owner) === (;)
        hd = Symbol("#barterm#lastn#d")
        @test CF.barterms(Covariance(:x, :d => double); hoistedby = owner) ===
              NamedTuple{(hd,)}((CF.BarTerm{:d}(double),))
        @test CF.barterms(RangeTotal(); hoistedby = owner) ===
              NamedTuple{(Symbol("#barterm#lastn#range"),)}((
            CF.BarTerm{:range}(rangeterm),))
        # A hoisted term is read as a column of its reserved name, not
        # evaluated again.
        types = NamedTuple{(:x, hd)}((Float64, Float64))
        st = CF.barwindow(Sum(:d => double), 2, types; hoistedby = owner)
        foreach(r -> CF.update!(st, NamedTuple{(:x, hd)}(r)),
            ((1.0, 10.0), (2.0, 20.0)))
        @test CF.value(st) == (d_sum = 30.0,)
        @test CF.value(CF.widenstate(st, types)) == (d_sum = 30.0,)
        # only the columns the window's summarizers read are stored
        @test fieldnames(eltype(st.ring.rows)) == (:d,)
        # and `:time`, which `inputcolumns` leaves out but `update!` may read
        tst = CF.barwindow(Sum(:d => double), 2, merge((time = Int,), types);
            hoistedby = owner)
        @test fieldnames(eltype(tst.ring.rows)) == (:time, :d)
        # A hoisted term is typed as an unhoisted one is, from inference over
        # the row (#103), and held to the same concreteness rule.
        mixed = r -> r.x > 2 ? r.x : 1
        chunk = DataFrame(x = [3.0, 4.0, 5.0])
        hoistin = CF.chunktypes(
            CF.terminput(chunk,
                NamedTuple{(hd,)}((CF.BarTerm{:d}(mixed),)), "summarize"),
        )
        hst = CF.barwindow(Sum(:d => mixed), 2, hoistin; hoistedby = owner)
        ust = CF.barwindow(Sum(:d => mixed), 2, (x = Float64,))
        @test eltype(hst.ring.rows) == eltype(ust.ring.rows) ==
              @NamedTuple{d::Union{Float64,Int64}}
        # A term inference gives up on is rejected on the first chunk,
        # whatever the data, as unhoisted it is at construction (#108); typed
        # by its values, it passed until a chunk mixed its branches.
        opaque = r -> Base.inferencebarrier(r.x > 2 ? r.x : 1)
        noinfer = ArgumentError(
            "barwindow row term :d must infer a concrete type, got Any")
        @test_throws noinfer CF.barwindow(Sum(:d => opaque), 2, (x = Float64,))
        # a hoisted column a widening (or a caller) types abstractly too
        @test_throws ArgumentError(
            "barwindow row term :d must infer a concrete type, got Any",
        ) CF.widenstate(hst, merge(hoistin, NamedTuple{(hd,)}((Any,))))
        for xs in ([3.0, 4.0, 5.0], [1.0, 4.0, 5.0])
            p = readtable(DataFrame(time = 1:3, x = xs))
            @test_throws noinfer load(Context(0, 10),
                p |> addsummarycolumns(LastN(Sum(:d => opaque), 2)))
            # a top-level term still falls back to its values
            @test eltype(
                DataFrame(load(Context(0, 10),
                    p |> addsummarycolumns(Sum(:d => opaque)))).d_sum,
            ) <: Real
        end
        # a widening re-checks the terms against the wider input
        @test_throws collides CF.widenstate(st, merge(types, (d = Float64,)))
        @test_throws notfound CF.barwindow(
            Sum(:d => double),
            2,
            (x = Float64,);
            hoistedby = owner,
        )
        # An input column of the term's own name is reported, not read as it,
        # hoisted or not: a wrapper that leaves out `rowterms` can't make the
        # window read the column in the term's place.
        @test_throws collides CF.barwindow(Sum(:d => double), 2, (x = Float64, d = Float64);
            hoistedby = owner)

        # The transforms evaluate the hoisted terms, keep only them, and agree
        # with an unhoisted window over every column.
        df = DataFrame(time = 1:8, k = repeat(["a", "b"], 4),
            x = Float64.(1:8), u = fill(missing, 8))
        hoisted = LastN(Mean(:d => double), 3)
        @test CF.rowterms(hoisted) === NamedTuple{(hd,)}((CF.BarTerm{:d}(double),))
        @test CF.inputcolumns(hoisted) == (:d, hd)
        expected = [missing, missing, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        for t in (addsummarycolumns(hoisted),
            addrollingcolumns((w = 10,), hoisted))
            out = DataFrame(load(Context(0, 10), readtable(df) |> t))
            @test isequal(out[!, end], expected)
            @test !("d" in names(out))
        end
        rec = Recording(hoisted, CF.inputcolumns(hoisted))
        out = DataFrame(
            load(Context(0, 10),
                readtable(df) |> addrollingcolumns((w = 10,), rec)),
        )
        @test !isempty(rec.seen)
        @test all(==((:time, hd)), rec.seen)
        @test isequal(out.w_lastn, expected)
        win = DataFrame(
            load(Context(0, 10),
                readtable(df) |>
                summarizewindows(clock(1), 10, [hoisted, LastNMean(:x, 3)])),
        )
        @test count(!ismissing, win.lastn) >= 3
        @test isequal(win.lastn, 2 .* win.x_lastnmean)
        @test only(
            eachrow(
                DataFrame(
                    load(Context(0, 10),
                        readtable(df) |> summarize(hoisted)))),
        ).lastn == 14.0

        # A top-level term of the name, or an input column of it, is reported;
        # a wrapper that leaves out `rowterms` is reported, input column or
        # not, rather than reading the column as the term.
        @test_throws collides load(Context(0, 10),
            readtable(df) |> addsummarycolumns([hoisted, Sum(:d => double)]))
        withd = DataFrame(time = 1:2, x = [1.0, 2.0], d = [0.0, 0.0])
        for s in (hoisted, Opaque(hoisted)),
            t in (addsummarycolumns(s),
                addrollingcolumns((w = 10,), s))

            @test_throws collides load(Context(0, 10), readtable(withd) |> t)
        end
        @test_throws notfound load(
            Context(0, 10),
            readtable(df) |> addsummarycolumns(Opaque(hoisted)),
        )
        # Two outer summarizers' windows each have a term `:d` of their own.
        triple = r -> 3r.x
        out = DataFrame(
            load(Context(0, 10),
                readtable(df) |> addrollingcolumns((w = 10,),
                    [hoisted, LastN{:other}(Mean(:d => triple), 3)])),
        )
        @test isequal(out.w_lastn, expected)
        @test isequal(out.w_other, 1.5 .* expected)
        @test CF.rowterms(LastN{:other}(Mean(:d => triple), 3)) ===
              NamedTuple{(Symbol("#barterm#other#d"),)}((CF.BarTerm{:d}(triple),))

        # The constructor's terms and the summarizer's own both count.
        protos, _, terms = CF.prototypes(
            CF.Summarizer[CF.Termed(hoisted, (; e = rangeterm))], Symbol[])
        @test terms === NamedTuple{(:e, hd)}((rangeterm, CF.BarTerm{:d}(double)))
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
