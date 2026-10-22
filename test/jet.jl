# Targeted JET checks: the kernels behind the function barriers must stay free
# of runtime dispatch. Whole-package analysis isn't used, since the untyped
# per-run setup fields (SummaryFold, RollingState, JoinState) are intended
# dynamism and would flood the report.

using JET

@testset "summarize kernels" begin
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Count(), Sum(:x), Mean(:x), Min(:x)]),
        Symbol[])
    outs = Val(requested)
    intypes = (time = Int, k = Int, x = Float64)
    states = CausalFrames.newstates(protos, intypes)
    nt = (time = [1, 1, 2], k = [1, 2, 1], x = [1.0, 2.0, 3.0])

    JET.@test_opt CausalFrames.foldall!(states, nt)
    JET.@test_opt CausalFrames.summaryvalues(states, outs)

    keynames = Val((:k,))
    groups = CausalFrames.newgroups(keynames, states, intypes)
    JET.@test_opt CausalFrames.foldgroups!(groups, states, nt, keynames)

    # Keys over `missing`-admitting columns: built in the table's key type and
    # ordered by `keyisless`, neither dispatching (issue #114).
    unames = Val((:k, :j))
    unt = (time = [1, 1, 2], k = Union{Missing,String}["a", missing, "a"],
        j = Union{Missing,Int}[1, 2, missing], x = [1.0, 2.0, 3.0])
    untypes = map(eltype, unt)
    ugroups = CausalFrames.newgroups(unames, states, untypes)
    JET.@test_opt CausalFrames.sortedgroups(ugroups)
    # On Julia 1.10, Base's NamedTuple `hash` dispatches from two `Union`
    # fields (issue #116), so the fold is checked over one key column there.
    # When 1.10 support is dropped, check both columns on every version.
    fnames = VERSION < v"1.11" ? Val((:k,)) : unames
    fgroups = CausalFrames.newgroups(fnames, states, untypes)
    JET.@test_opt CausalFrames.foldgroups!(fgroups, states, unt, fnames)

    JET.@test_opt CausalFrames.foldgroups!(states, states, nt, Val(()))
    JET.@test_opt CausalFrames.foldcycles!(states, states, nt, nothing, Val(()),
        protos, outs)
    JET.@test_opt CausalFrames.foldcycles!(groups, states, nt, nothing, keynames,
        protos, outs)
    JET.@test_opt CausalFrames.flushcycle(states, states, 1, Val(()), protos, outs)
    JET.@test_opt CausalFrames.flushcycle(groups, states, 1, keynames, protos, outs)
    JET.@test_opt CausalFrames.freshall!(states)
    JET.@test_opt CausalFrames.foldrunning!(states, states, nt, 3, Val(()), outs)
    JET.@test_opt CausalFrames.foldrunning!(groups, states, nt, 3, keynames, outs)
end

@testset "wide rows" begin
    # Past 32 columns, Base's tuple `map` falls back to dispatch (issue #98).
    nt = NamedTuple{Tuple(Symbol.(:c, 1:40))}(Tuple(rand(3) for _ in 1:40))
    V = CausalFrames.storerowtype(map(eltype, nt))
    JET.@test_opt CausalFrames.rowat(V, nt, 2)
end

@testset "dependent summarizer emission" begin
    # A dependent summarizer's `value` runs per emitted row under
    # addsummarycolumns and addrollingcolumns, so it must be dispatch-free too.
    # `@inferred` isn't enough: it checks only the return type, and a body that
    # derives an element type from `typeof` of the dependency values goes
    # dynamic while still returning a concrete type.
    lr = (time = Int, x = Float64, z = Float64, y = Float64)
    st = CausalFrames.fresh(LinearRegression(:x, :y), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_x_comoment = 10.0, y_y_comoment = 20.0,
            x_y_comoment = 10.0, x_sum = 15.0, y_sum = 30.0))
    st = CausalFrames.fresh(LinearRegression(:x, :y; intercept = false), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_sumpower_2 = 55.0, y_sumpower_2 = 200.0,
            x_y_dotproduct = 100.0))
    st = CausalFrames.fresh(LinearRegression([:x, :z], :y), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_x_comoment = 10.0, z_z_comoment = 10.0,
            y_y_comoment = 20.0, x_z_comoment = 5.0, x_y_comoment = 10.0,
            y_z_comoment = 0.0, x_sum = 15.0, z_sum = 15.0, y_sum = 30.0))

    # The Missing-admitting variants, where the early return is live and the
    # dependency values are Union-typed. Both arities are checked, since a
    # one-element tuple of Union values union-splits but a K >= 2 tuple may not.
    MF = Union{Missing,Float64}
    st = CausalFrames.fresh(LinearRegression(:x, :y),
        (time = Int, x = MF, y = Float64))
    JET.@test_opt CausalFrames.value(st,
        NamedTuple{(:count, :x_x_comoment, :y_y_comoment, :x_y_comoment,
            :x_sum, :y_sum),
            Tuple{Int,MF,Float64,MF,MF,Float64}}((5, 10.0, 20.0, 10.0, 15.0,
            30.0)))
    for T in (MF, Union{Missing,Int})
        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers([LinearRegression([:x, :z], :y)]),
            Symbol[])
        states = CausalFrames.newstates(protos,
            (time = Int, x = T, z = T, y = T))
        JET.@test_opt CausalFrames.summaryvalues(states, Val(requested))
    end

    # the rename a non-canonical symmetric summarizer folds through
    st = CausalFrames.fresh(DotProduct(:y, :x), (time = Int, x = Int, y = Int))
    JET.@test_opt CausalFrames.value(st, (x_y_dotproduct = 19,))

    # and the whole accumulate-then-project fold with a regression in the set
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([LinearRegression(:x, :y), Count()]),
        Symbol[])
    states = CausalFrames.newstates(protos,
        (time = Int, x = Float64, y = Float64))
    nt = (time = [1, 1, 2], x = [1.0, 2.0, 3.0], y = [3.0, 5.0, 7.0])
    JET.@test_opt CausalFrames.foldall!(states, nt)
    JET.@test_opt CausalFrames.summaryvalues(states, Val(requested))
end

@testset "SumPower term specialization" begin
    # the specialized terms must be as dispatch-free as the general one
    for n in (1, 2, 3)
        st = CausalFrames.fresh(SumPower(:x, n), (time = Int, x = Float64))
        JET.@test_opt CausalFrames.update!(st, (time = 1, x = 0.5))
        JET.@test_opt CausalFrames.downdate!(st, (time = 1, x = 0.5))
    end
end

@testset "compensated accumulators" begin
    st = CausalFrames.fresh(Sum(:x), (time = Int, x = Float64))
    row = (time = 1, x = 0.5)
    JET.@test_opt CausalFrames.update!(st, row)
    JET.@test_opt CausalFrames.downdate!(st, row)
    JET.@test_opt CausalFrames.value(st)
end

@testset "co-moment accumulator" begin
    # Variance's and Covariance's accumulator, over every storage it takes:
    # exact integers, compensated floats (with recentring) and the
    # missing-counting flag
    for T in (Int, Float64, Union{Missing,Int}, Union{Missing,Float64}),
        s in (CausalFrames.CoMoment{:x,:x}(), CausalFrames.CoMoment{:x,:y}())

        st = CausalFrames.fresh(s, (time = Int, x = T, y = T))
        row = (time = 1, x = one(nonmissingtype(T)), y = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.update!(st, row)
        JET.@test_opt CausalFrames.downdate!(st, row)
        JET.@test_opt CausalFrames.value(st)
        JET.@test_opt CausalFrames.combine!(st, st, CausalFrames.fresh(st))
    end
    st = CausalFrames.fresh(Covariance(:y, :x), (time = Int, x = Float64, y = Float64))
    JET.@test_opt CausalFrames.value(st, (count = 3, x_y_comoment = -3.0))
end

@testset "missing-counting accumulators" begin
    # the counting states over a Union{Missing,_} column must stay dispatch-free
    for T in (Union{Missing,Int}, Union{Missing,Float64})
        st = CausalFrames.fresh(Sum(:x), (time = Int, x = T))
        row = (time = 1, x = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.update!(st, row)
        JET.@test_opt CausalFrames.downdate!(st, row)
        JET.@test_opt CausalFrames.value(st)
    end
end

@testset "order statistics" begin
    # the sorted accumulator's fold and its dependents' emission, over a plain
    # and a Missing-admitting column, and its merge. A window slides the
    # windowed states (Last's ordinary TrackState has no downdate!), so the
    # slide is checked over those and the rest over the ordinary ones.
    for T in (Int, Union{Missing,Float64})
        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers(
                [Quantile(:x, [0.1, 0.5]), Median(:x), PercentRank(:x), MeanAbsDev(:x),
                Quantile(:x, 0.9; interpolation = :nearestrank)]), Symbol[])
        intypes = (time = Int, x = T)
        states = CausalFrames.newstates(protos, intypes)
        windowed = map(s -> CausalFrames.freshwindowed(s, intypes), protos)
        row = (time = 1, x = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.updateall!(states, row)
        CausalFrames.updateall!(states, row)
        JET.@test_opt CausalFrames.summaryvalues(states, Val(requested))
        JET.@test_opt CausalFrames.updateall!(windowed, row)
        CausalFrames.updateall!(windowed, row)
        JET.@test_opt CausalFrames.downdateall!(windowed, row)
        JET.@test_opt CausalFrames.summaryvalues(windowed, Val(requested))
        st = first(states)
        JET.@test_opt CausalFrames.combine!(st, st, CausalFrames.fresh(st))
    end
end

@testset "monoid queues" begin
    protos, _ = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Min(:x), Max(:x)]), Symbol[])
    states = CausalFrames.newstates(protos, (time = Int, x = Float64))
    row = (time = 1, x = 1.0)
    Q = CausalFrames.queuestype(states, typeof(row), Int)
    mq = CausalFrames.newqueues(Q, states, 2)
    lookbacks = (3, Bars(2))
    JET.@test_opt CausalFrames.queuepush!(mq, row, lookbacks)
    CausalFrames.queuepush!(mq, row, lookbacks)
    JET.@test_opt CausalFrames.queueexpire!(mq, 2, 1, lookbacks...)
    CausalFrames.queueexpire!(mq, 2, 1, lookbacks...)
    JET.@test_opt CausalFrames.queuestates(mq, 1)
    CausalFrames.queuestates(mq, 1)
    JET.@test_opt CausalFrames.flipqueue!(mq.queues[2], mq.rows, mq.dropped, 1)
end

@testset "intervalize kernels" begin
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Count(), Sum(:x), Mean(:x)]), Symbol[])
    outs = Val(requested)
    intypes = (time = Int, k = Int, x = Float64)
    states = CausalFrames.newstates(protos, intypes)
    nt = (time = [1, 3, 6], k = [1, 2, 1], x = [1.0, 2.0, 3.0])
    bounds = [0, 5, 10]

    # the keyless grid: the dense kernels over the empty key
    nks = CausalFrames.nokeyset("intervalize")
    ndg = CausalFrames.newgroups(nks, states, intypes)
    JET.@test_opt CausalFrames.foldintervals!(ndg, states, nt, bounds, 2, nks,
        true, protos, outs)
    JET.@test_opt CausalFrames.flushintervals!(ndg, states, bounds, 2, 10, nks,
        true, protos, outs)

    keynames = Val((:k,))
    groups = CausalFrames.newgroups(keynames, states, intypes)
    JET.@test_opt CausalFrames.foldintervals!(groups, states, nt, bounds, 2,
        keynames, true, protos, outs)
    JET.@test_opt CausalFrames.flushintervals!(groups, states, bounds, 2, 10,
        keynames, true, protos, outs)

    # the declared-key (dense) kernels, shared with summarizecycles; the keyset
    # is declared Float64 against the Int key column, so the slot lookup crosses
    # key types as a real declaration may
    ks = CausalFrames.tokeyset([2.0, 1.0], [:k], "intervalize")
    dg = CausalFrames.newgroups(ks, states, intypes)
    JET.@test_opt CausalFrames.foldintervals!(dg, states, nt, bounds, 2, ks,
        true, protos, outs)
    JET.@test_opt CausalFrames.flushintervals!(dg, states, bounds, 2, 10, ks,
        true, protos, outs)
    DRT, emptyrow = CausalFrames.densetypes(Int, states, protos, ks, outs)
    JET.@test_opt CausalFrames.closedense!(DRT[], dg, ks, 5, outs, emptyrow)
    JET.@test_opt CausalFrames.foldcycles!(dg, states, nt, nothing, ks, protos,
        outs)
    JET.@test_opt CausalFrames.flushcycle(dg, states, 1, ks, protos, outs)
end

@testset "asofjoin kernel" begin
    # a String field makes V non-isbits, the case the slot store exists for
    V = typeof((time = 1, sym = "a", y = 1.0))
    K = typeof((sym = "a",))
    index = Dict{K,Int}()
    slots = V[]
    matches = Vector{V}(undef, 1)
    found = [false]
    lnt = (time = [1], sym = ["a"])
    rnt = (time = [0], sym = ["a"], y = [2.0])
    JET.@test_opt CausalFrames.joinsegment!(matches, found, index, slots, lnt,
        1, rnt, 1, true, Val((:sym,)), <=, nothing)
    JET.@test_opt CausalFrames.matchcolumn(matches, found, Val(:y))
end

@testset "lookupjoin kernels" begin
    # a String key, which only the Int row vector keeps from boxing per row, and
    # the Missing-widening gather over a non-isbits and an isbits column
    K = typeof((sym = "a",))
    index = Dict{K,Int}((sym = "a",) => 1)
    nt = (time = [1, 2], sym = ["a", "b"])
    JET.@test_opt CausalFrames.lookuprows!(Vector{Int}(undef, 2), index, nt,
        Val((:sym,)))
    JET.@test_opt CausalFrames.gathermissing(["x"], [1, 0])
    JET.@test_opt CausalFrames.gathermissing([1.0], [1, 0])

    # the index build over a table past 32 columns, where a tuple `map` over
    # every column stops inferring and each insert would dispatch
    wide = (; sym = ["a", "b"], (Symbol(:v, j) => [1.0, 2.0] for j in 1:39)...)
    vn = Tuple(Symbol(:v, j) for j in 1:39)
    JET.@test_opt CausalFrames.buildlookup(wide, Val((:sym,)), Val(vn), [:sym],
        CausalFrames.MissingOnUnmatched(), nothing, collect(vn))
end

@testset "lastrow kernel" begin
    # the same non-isbits V the slot store exists for
    V = typeof((time = 1, sym = "a", y = 1.0))
    K = typeof((sym = "a",))
    index = Dict{K,Int}()
    slots = V[]
    nt = (time = [1, 2], sym = ["a", "b"], y = [1.0, 2.0])
    JET.@test_opt CausalFrames.lastsegment!(index, slots, nt, Val((:sym,)))
end

@testset "sortcycles kernel" begin
    # a mixed isbits/String key tuple over column views (what the operator
    # passes), a tuple-valued function key, and both orderings — the Ordering
    # is a type, so neither direction may split
    rows = 2:7
    times = view([0, 1, 1, 1, 2, 3, 3], rows)
    keys = (view([0, 2, 1, 2, 9, 1, 1], rows),
        view(["", "b", "a", "c", "z", "b", "a"], rows))
    for order in (Base.Order.Forward, Base.Order.Reverse)
        JET.@test_opt CausalFrames.cycleperm!(Int[], keys, times, order)
    end
    fkeys = ([(-2, "b"), (-1, "a"), (-2, "c"), (-9, "z"), (-1, "b"), (-1, "a")],)
    JET.@test_opt CausalFrames.cycleperm!(Int[], fkeys, times, Base.Order.Forward)
end

# The tier combinations the window kernels specialize on: each tier alone, and
# all of them with a dependent spanning them (TierSpan, test/fixtures.jl), over
# a String key so the key is non-isbits.
const JETTIERSETS = (
    [Count(), Sum(:x), Mean(:x), Last(:x), Min(:x),
        CountDistinct(:x), AgeWeightedSum(:x), Std(:x)], [Product(:x)], [PlainSum(:x)],
    [TierSpan(:x), Mean(:x), First(:x)])

@testset "addrollingcolumns kernel" begin
    types = (time = Int, k = String, x = Float64)
    lnt = (time = [1, 3, 6], k = ["a", "b", "a"], x = [0.0, 0.0, 0.0])
    snt = (time = [1, 2, 6], k = ["a", "b", "a"], x = [1.0, 2.0, 3.0])
    # time windows, a Bars window beside one (their value types differ), and
    # Bars windows alone (no buffer)
    for ss in JETTIERSETS, kn in (Val((:k,)), Val(())),
        lookbacks in ((1, 5), (Bars(2), 5), (Bars(2), Bars(3)))

        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers(ss), Symbol[])
        outs = Val(requested)
        tg = CausalFrames.tiering(protos, types)
        tiers = CausalFrames.rolltiers(tg, types, kn, lookbacks, nothing)
        rs = CausalFrames.RollingState(())
        cfg = CausalFrames.RollingConfig((:a, :b), lookbacks, Symbol[], kn,
            protos, outs, Symbol[], (;), nothing)
        CausalFrames.setvaltype!(rs, cfg, tg)
        wins = map((lb, e, T) -> (lb, e, Vector{T}(undef, 3)), lookbacks,
            rs.emptyrows, rs.valtypes)
        JET.@test_opt CausalFrames.rollsegment!(wins, tiers, lnt, 1, snt, 1,
            true, lookbacks, kn, outs)
        JET.@test_opt CausalFrames.leadinsegment!(tiers, snt, 1, 6, lookbacks,
            kn)
    end
end

@testset "summarizewindows kernels" begin
    # keyed (with the vanish-row merge), keyless (the grid over the empty key),
    # and dense (the declared emission and its keyset checks)
    types = (time = Int, k = String, x = Float64)
    nt = (time = [1, 3, 6], k = ["a", "b", "a"], x = [1.0, 2.0, 3.0])
    ticks = [0, 5, 10]
    declared = CausalFrames.tokeyset(["b", "a"], [:k], "summarizewindows")
    for ss in JETTIERSETS,
        (kc, ks) in (([:k], nothing),
            (Symbol[], CausalFrames.nokeyset("summarizewindows")), ([:k], declared))

        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers(ss), kc)
        outs = Val(requested)
        kn = Val(Tuple(kc))
        cfg = CausalFrames.WindowConfig(kc, kn, 5, protos, outs, ks, (;), nothing)
        st = CausalFrames.WindowState{Int}(
            CausalFrames.IntervalCursor{Int}(clock(5).run(Context(0, 10))), cfg)
        CausalFrames.preparewindows!(st, cfg, types)
        RT, emptyrow = CausalFrames.windowtypes(st, cfg)
        JET.@test_opt CausalFrames.windowrows!(RT[], st.tiers, 1, nt, ticks,
            st.prevkeys, 5, kn, outs, emptyrow, ks)
        JET.@test_opt CausalFrames.flushwindows!(RT[], st.tiers, 1, ticks,
            st.prevkeys, 5, kn, outs, emptyrow, ks)
    end
end

@testset "windowed states and the age-weighted sum" begin
    for T in (Float64, Int, Union{Missing,Float64}),
        s in (Min(:x), First(:x),
            Last(:x), CountDistinct(:x), AgeWeightedSum(:x), MaxIndex(:x),
            MinWithIndex(:x))

        st = CausalFrames.freshwindowed(s, (time = Int, x = T))
        row = (time = 1, x = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.update!(st, row)
        CausalFrames.update!(st, row)
        JET.@test_opt CausalFrames.value(st)
        JET.@test_opt CausalFrames.downdate!(st, row)
    end
    st = CausalFrames.fresh(AgeWeightedSum(:x), (time = Int, x = Float64))
    JET.@test_opt CausalFrames.combine!(st, st, st)
    # the arg-extremes' ordinary state, which the plain folds and the queues use
    for T in (Float64, Union{Missing,Float64}), s in (MaxIndex(:x), MinWithIndex(:x))
        st = CausalFrames.fresh(s, (time = Int, x = T))
        row = (time = 1, x = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.update!(st, row)
        CausalFrames.update!(st, row)
        JET.@test_opt CausalFrames.value(st)
        JET.@test_opt CausalFrames.combine!(st, st, CausalFrames.fresh(st))
    end
end

# A FitModel fold is a typed push per column (the fit itself is opaque and runs
# once per emitted summary, not per row), and once the buffers have capacity a
# fold allocates nothing — the property `fresh!` keeping capacity exists for.
foldallocs(st, row) = @allocated CausalFrames.update!(st, row)
@testset "FitModel fold" begin
    st = CausalFrames.fresh(FitModel(ToyOLS(), [:x, :z], :y),
        (time = Int, x = Float64, z = Int, y = Float64))
    row = (time = 1, x = 1.0, z = 2, y = 3.0)
    JET.@test_opt CausalFrames.update!(st, row)
    JET.@test_opt CausalFrames.fresh!(st)
    for _ in 1:100
        CausalFrames.update!(st, row)
    end
    CausalFrames.fresh!(st)
    foldallocs(st, row)
    @test foldallocs(st, row) == 0
end

@testset "forwardfill kernels" begin
    # the same non-isbits carried value the mutable cells exist for
    C = CausalFrames.FillCell{String,Int}
    times = [1, 2]
    incol = Union{Missing,String}["a", missing]
    groups = ((incol, Vector{Union{Missing,String}}(undef, 2)),)
    JET.@test_opt CausalFrames.fillkeyless!(times, groups, (C(),), 2)

    NT = typeof((v = C(),))
    store = Dict{typeof((k = "a",)),NT}()
    nt = (time = times, k = ["a", "a"], v = incol)
    JET.@test_opt CausalFrames.fillkeyed!(times, groups, nt, Val((:k,)), store, 2)
end

@testset "settime kernel" begin
    # the per-row causality check, behind its function barrier
    JET.@test_opt CausalFrames.checkforward([1, 2], [3, 4], "settime")
end

@testset "readtable kernels" begin
    # the generic path's per-partition barrier, in both its range and its
    # stable-sort forms, the missing-time filter every source shares, and the
    # typed copy of the kept times
    JET.@test_opt CausalFrames.tablerows([1, 2, 3], true, false, false, nothing, 0, 3)
    JET.@test_opt CausalFrames.tablerows([3, 1, 2], true, true, true, nothing, 0, 3)
    JET.@test_opt CausalFrames.presentrows([1, missing, 3], true, "table", "")
    JET.@test_opt CausalFrames.presentrows([1, 2, 3], false, "CSV file", "x.csv")
    JET.@test_opt CausalFrames.windowbounds([1, 2, 3], true, 1, 3)
    JET.@test_opt CausalFrames.copytimes(Float64, [1, 2, 3], 1:2)
    JET.@test_opt CausalFrames.copytimes(Int, [1, 2, 3], [3, 1])
end

@testset "merge winner selection" begin
    # the one per-block loop over the cursors: the ordering comparisons must
    # stay on the concrete time type
    cursors = [CausalFrames.MergeCursor{Int}(i, nothing) for i in 1:2]
    for cur in cursors
        cur.chunk = DataFrame(time = [cur.index])
        cur.times = [cur.index]
    end
    JET.@test_opt CausalFrames.pickwinner(cursors)
end

@testset "row term columns" begin
    # A term column is computed per chunk behind `termvalues`' barrier, which
    # must specialize on the term function and the column table.
    nt = (time = [1, 2], high = [5.0, 6.0], low = [4.0, 4.5])
    JET.@test_opt CausalFrames.termvalues(rangeterm, nt)
    # a Union-typed term too, whose concreteness check must fold
    JET.@test_opt CausalFrames.termvalues(r -> r.high > 5 ? r.high : 1, nt)
    # over rows with `String` and `missing`-admitting columns, past the 32
    # columns where a tuple `map` stops inferring
    wide = merge(nt, (k = ["a", "b"],),
        NamedTuple{Tuple(Symbol.(:c, 1:40))}(
            ntuple(_ -> Union{Missing,Float64}[1.0, missing], 40)))
    JET.@test_opt CausalFrames.termvalues(rangeterm, wide)
    JET.@test_opt CausalFrames.termvalues(r -> r.c40, wide)
    # a hoisted count-window term, whose type check must fold
    JET.@test_opt CausalFrames.termvalues(CausalFrames.BarTerm{:range}(rangeterm), wide)
    protos, requested, terms = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Sum(:range => rangeterm)]), Symbol[])
    tnt = merge(nt, (; range = CausalFrames.termvalues(rangeterm, nt)))
    states = CausalFrames.newstates(protos, CausalFrames.chunktypes(tnt))
    JET.@test_opt CausalFrames.foldall!(states, tnt)
end

@testset "barwindow, alone and embedded in an outer state" begin
    # The count-window helper exists to live inside another state, whose
    # update!/value the transforms' own tuple folds call: its per-row path must
    # stay dispatch-free there too, where a shared fold would hit inference's
    # recursion limit.
    for s in (Std(:x), Covariance(:x, :d => r -> 2r.x), Product(:x),
        Sum(:d => r -> 2r.x))
        st = CausalFrames.barwindow(s, 5, (x = Float64,))
        JET.@test_opt CausalFrames.update!(st, (x = 1.0,))
        JET.@test_opt CausalFrames.value(st)
    end
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([LastNMean(:x, 3), Sum(:x)]), Symbol[])
    outs = Val(requested)
    states = CausalFrames.newstates(protos, (time = Int, x = Float64))
    nt = (time = [1, 2, 3, 4], x = [1.0, 2.0, 3.0, 4.0])
    JET.@test_opt CausalFrames.foldall!(states, nt)
    JET.@test_opt CausalFrames.summaryvalues(states, outs)
    JET.@test_opt CausalFrames.foldrunning!(states, states, nt, 4, Val(()), outs)
    # with the window's row term hoisted to the transform
    protos, requested, terms = CausalFrames.prototypes(
        CausalFrames.tosummarizers([LastN(Covariance(:x, :range => rangeterm), 3)]),
        Symbol[])
    outs = Val(requested)
    tnt = (time = [1, 2], x = [1.0, 2.0], high = [3.0, 4.0], low = [1.0, 1.5])
    tnt = merge(tnt, map(f -> CausalFrames.termvalues(f, tnt), terms))
    states = CausalFrames.newstates(protos, CausalFrames.chunktypes(tnt))
    JET.@test_opt CausalFrames.foldall!(states, tnt)
    JET.@test_opt CausalFrames.summaryvalues(states, outs)
    JET.@test_opt CausalFrames.foldrunning!(states, states, tnt, 2, Val(()), outs)
end
