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

    JET.@test_opt CausalFrames.foldcycles!(states, states, nt, nothing, outs)
    JET.@test_opt CausalFrames.foldrunning!(states, nt, 3, outs)
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
    JET.@test_opt CausalFrames.treequery(tr, states, 1, 1)
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

    JET.@test_opt CausalFrames.foldintervals!(states, states, protos, nt,
        bounds, 2, false, true, outs)
    JET.@test_opt CausalFrames.flushintervals!(states, states, protos, bounds,
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
    V = typeof((time = 1, y = 1.0))
    store = Dict{NamedTuple{(),Tuple{}},V}()
    matches = Union{Missing,V}[missing]
    lnt = (time = [1],)
    rnt = (time = [0], y = [2.0])
    JET.@test_opt CausalFrames.joinsegment!(matches, store, lnt, 1, rnt, 1,
        true, Val(()), <=, nothing)
end
