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

@testset "segment tree" begin
    protos, _ = CausalFrames.prototypes(
        CausalFrames.tosummarizers([Min(:x), Max(:x)]), Symbol[])
    states = CausalFrames.newstates(protos, (time = Int, x = Float64))
    row = (time = 1, x = 1.0)
    tr = CausalFrames.newsegtree(states, typeof(row), Int)
    JET.@test_opt CausalFrames.treepush!(tr, states, row)
    CausalFrames.treepush!(tr, states, row)
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

@testset "settime kernel" begin
    # the per-row causality check, behind its function barrier
    JET.@test_opt CausalFrames.checkforward([1, 2], [3, 4], "settime")
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
