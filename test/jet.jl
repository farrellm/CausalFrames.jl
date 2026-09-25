# Targeted JET checks: the folding kernels sit behind function barriers and
# must stay free of runtime dispatch end to end. Whole-package analysis is
# deliberately not used — the dynamically typed per-run setup fields
# (SummaryFold, RollingState, AsofJoinState) are intended dynamism and would
# flood a package-level report.

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
    groups = CausalFrames.newgroups(states, nt, keynames)
    JET.@test_opt CausalFrames.foldgroups!(groups, states, nt, keynames)

    JET.@test_opt CausalFrames.foldcycles!(states, nt, nothing, outs)
    JET.@test_opt CausalFrames.freshall!(states)
    JET.@test_opt CausalFrames.foldrunning!(states, nt, 3, outs)
end

@testset "dependent summarizer emission" begin
    # A dependent summarizer's `value` runs per emitted row on the
    # addsummarycolumns and addrollingcolumns paths, so it has to stay
    # dispatch-free too. `@inferred` is not enough on its own: it proves the
    # *return* type is concrete while saying nothing about the body, and the
    # missing-admitting regression passed it while every line inside went
    # dynamic — because the element type was derived from `typeof` of the
    # dependency values, which is value-dependent and so not a constant.
    lr = (time = Int, x = Float64, z = Float64, y = Float64)
    st = CausalFrames.fresh(LinearRegression(:x, :y), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_sumpower_2 = 55.0, y_sumpower_2 = 200.0,
            x_y_dotproduct = 100.0, x_sum = 15.0, y_sum = 30.0))
    st = CausalFrames.fresh(LinearRegression(:x, :y; intercept = false), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_sumpower_2 = 55.0, y_sumpower_2 = 200.0,
            x_y_dotproduct = 100.0))
    st = CausalFrames.fresh(LinearRegression([:x, :z], :y), lr)
    JET.@test_opt CausalFrames.value(st,
        (count = 5, x_sumpower_2 = 55.0, z_sumpower_2 = 55.0,
            y_sumpower_2 = 200.0, x_z_dotproduct = 50.0,
            x_y_dotproduct = 100.0, y_z_dotproduct = 90.0,
            x_sum = 15.0, z_sum = 15.0, y_sum = 30.0))

    # The Missing-admitting variants, where the early-return branch is live and
    # the dependency values are Union-typed. Both arities are checked on
    # purpose: a one-element tuple of Union values union-splits and looks fine,
    # so K = 1 stayed clean while the K >= 2 cross-product tuple went dynamic.
    MF = Union{Missing,Float64}
    st = CausalFrames.fresh(LinearRegression(:x, :y),
        (time = Int, x = MF, y = Float64))
    JET.@test_opt CausalFrames.value(st,
        NamedTuple{(:count, :x_sumpower_2, :y_sumpower_2, :x_y_dotproduct,
            :x_sum, :y_sum),
            Tuple{Int,MF,Float64,MF,MF,Float64}}((5, 55.0, 200.0, 100.0, 15.0,
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
                [Quantile(:x, [0.1, 0.5]), Median(:x), PercentRank(:x),
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

@testset "segment tree" begin
    protos, _ = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Min(:x), Max(:x)]), Symbol[])
    states = CausalFrames.newstates(protos, (time = Int, x = Float64))
    row = (time = 1, x = 1.0)
    tr = CausalFrames.newsegtree(states, typeof(row), Int)
    JET.@test_opt CausalFrames.treepush!(tr, states, row)
    CausalFrames.treepush!(tr, states, row)
    JET.@test_opt CausalFrames.treeappend!(tr, states, row)
    CausalFrames.treeappend!(tr, states, row)
    JET.@test_opt CausalFrames.treesync!(tr)
    CausalFrames.treesync!(tr)
    JET.@test_opt CausalFrames.treequery(tr, 1, 1)
    JET.@test_opt CausalFrames.windowstart(tr.times, tr.head, 1, 0)
end

@testset "intervalize kernels" begin
    protos, requested = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Count(), Sum(:x), Mean(:x)]), Symbol[])
    outs = Val(requested)
    intypes = (time = Int, k = Int, x = Float64)
    states = CausalFrames.newstates(protos, intypes)
    nt = (time = [1, 3, 6], k = [1, 2, 1], x = [1.0, 2.0, 3.0])
    bounds = [0, 5, 10]

    JET.@test_opt CausalFrames.foldintervals!(states, protos, nt,
        bounds, 2, false, true, outs)
    JET.@test_opt CausalFrames.flushintervals!(states, protos, bounds,
        2, false, 10, true, outs)

    keynames = Val((:k,))
    groups = CausalFrames.newgroups(states, nt, keynames)
    JET.@test_opt CausalFrames.foldintervalsgrouped!(groups, states, nt, bounds,
        2, keynames, true, outs)
    RT = CausalFrames.rowtype(Int, keytype(groups), valtype(groups), outs)
    JET.@test_opt CausalFrames.flushintervalsgrouped!(groups, bounds, 2, RT, 10,
        true, outs)

    # the declared-key (dense) kernels, shared with summarizecycles; the keyset
    # is declared Float64 against the Int key column, so the slot lookup crosses
    # key types as a real declaration may
    ks = CausalFrames.tokeyset([2.0, 1.0], [:k], "intervalize")
    dg = CausalFrames.densegroups(states, 2)
    DRT, emptyrow = CausalFrames.densetypes(Int, states, protos, ks, outs)
    JET.@test_opt CausalFrames.foldintervalsdense!(DRT[], dg, ks, nt, bounds, 2,
        keynames, true, outs, emptyrow)
    JET.@test_opt CausalFrames.flushintervalsdense!(DRT[], dg, ks, bounds, 2, 10,
        true, outs, emptyrow)
    JET.@test_opt CausalFrames.closedense!(DRT[], dg, ks, 5, outs, emptyrow)
    JET.@test_opt CausalFrames.foldcyclesdense!(DRT[], dg, ks, nt, nothing,
        keynames, outs, emptyrow)
end

@testset "asofjoin kernel" begin
    # a String field makes V non-isbits, which is the case the index/slots
    # store exists for: a Dict of rows could only answer as Union{Nothing,V},
    # boxing once per left row
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
end

@testset "lastrow kernel" begin
    # the same non-isbits V the asofjoin store exists for: a Dict{K,V} would
    # answer every lookup as Union{Nothing,V} and box it once per row
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
const JETTIERSETS = ([Count(), Sum(:x), Mean(:x), Last(:x), Min(:x),
    CountDistinct(:x), AgeWeightedSum(:x)], [Product(:x)], [PlainSum(:x)],
    [TierSpan(:x), Mean(:x), First(:x)])

@testset "addrollingcolumns kernel" begin
    types = (time = Int, k = String, x = Float64)
    lnt = (time = [1, 3, 6], k = ["a", "b", "a"], x = [0.0, 0.0, 0.0])
    snt = (time = [1, 2, 6], k = ["a", "b", "a"], x = [1.0, 2.0, 3.0])
    for ss in JETTIERSETS, kn in (Val((:k,)), Val(()))
        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers(ss), Symbol[])
        outs = Val(requested)
        tg = CausalFrames.tiering(protos, types)
        tiers = CausalFrames.rolltiers(tg, types, kn, 2, nothing)
        V = CausalFrames.tieredvaluetype(tg, protos, outs)
        emptyrow = convert(V, CausalFrames.emptyvalues(protos, outs))
        vals = (Vector{V}(undef, 3), Vector{V}(undef, 3))
        JET.@test_opt CausalFrames.rollsegment!(vals, tiers, lnt, 1, snt, 1,
            true, (1, 5), kn, outs, emptyrow)
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
        (kc, grid, ks) in (([:k], Val(false), nothing),
            (Symbol[], Val(true), nothing), ([:k], Val(false), declared))

        protos, requested = CausalFrames.prototypes(
            CausalFrames.tosummarizers(ss), kc)
        outs = Val(requested)
        kn = Val(Tuple(kc))
        cfg = CausalFrames.WindowConfig(kc, kn, 5, protos, outs, grid, ks)
        st = CausalFrames.WindowState{Int}(
            CausalFrames.IntervalCursor{Int}(clock(5).run(Context(0, 10))))
        CausalFrames.preparewindows!(st, cfg, types)
        RT, emptyrow = CausalFrames.windowtypes(st, cfg)
        JET.@test_opt CausalFrames.windowrows!(RT[], st.tiers, 1, nt, ticks,
            st.prevkeys, 5, kn, outs, emptyrow, grid, ks)
        JET.@test_opt CausalFrames.flushwindows!(RT[], st.tiers, 1, ticks,
            st.prevkeys, 5, kn, outs, emptyrow, grid, ks)
    end
end

@testset "windowed states and the age-weighted sum" begin
    for T in (Float64, Int, Union{Missing,Float64}),
        s in (Min(:x), First(:x),
            Last(:x), CountDistinct(:x), AgeWeightedSum(:x))

        st = CausalFrames.freshwindowed(s, (time = Int, x = T))
        row = (time = 1, x = one(nonmissingtype(T)))
        JET.@test_opt CausalFrames.update!(st, row)
        CausalFrames.update!(st, row)
        JET.@test_opt CausalFrames.value(st)
        JET.@test_opt CausalFrames.downdate!(st, row)
    end
    st = CausalFrames.fresh(AgeWeightedSum(:x), (time = Int, x = Float64))
    JET.@test_opt CausalFrames.combine!(st, st, st)
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
