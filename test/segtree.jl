@testset "segment tree" begin
    intypes = (time = Int64, x = Int64)
    protos(names...) = map(s -> CausalFrames.fresh(s, intypes), names)
    R = @NamedTuple{time::Int64, x::Int64}

    # a state tuple mixing an invertible accumulator, order-insensitive and
    # order-sensitive trackers, and the monoid-only product
    ss = (Sum(:x), Min(:x), Last(:x), Product(:x))
    stateprotos = protos(ss...)

    naive(rows) = begin
        sts = map(CausalFrames.fresh, stateprotos)
        foreach(r -> foreach(st -> CausalFrames.update!(st, r), sts), rows)
        CausalFrames.summaryvalues(sts, Val((:x_sum, :x_min, :x_last,
            :x_product)))
    end
    queryvals(tr, lo, hi) =
        CausalFrames.summaryvalues(
            CausalFrames.treequery(tr, stateprotos, lo, hi),
            Val((:x_sum, :x_min, :x_last, :x_product)))

    # pseudo-random pushes cross-checked against a naive re-fold, across
    # several capacity-doubling rebuilds; products of ±1/±2 stay exact
    steps = lcgsequence(7, 100, 3)      # deliberate ties (zero steps)
    xs = map(v -> (-2, -1, 1, 2)[v+1], lcgsequence(11, 100, 4))
    picks = lcgsequence(13, 200, 2^30)
    tr = CausalFrames.newsegtree(stateprotos, R, Int64)
    rows = R[]
    t = 0
    for i in 1:100
        t += steps[i]
        row = (time = t, x = xs[i])
        push!(rows, row)
        CausalFrames.treepush!(tr, stateprotos, row)
        lo = 1 + picks[2i-1] % i
        hi = lo + picks[2i] % (i - lo + 1)
        @test queryvals(tr, lo, hi) == naive(rows[lo:hi])
        @test queryvals(tr, i, i) == naive(rows[i:i])
        @test queryvals(tr, 1, i) == naive(rows)
    end
    @test @inferred(CausalFrames.treequery(tr, stateprotos, 1,
        length(rows))) isa Tuple

    # advancing head drops the expired prefix at the next full-capacity
    # rebuild, and queries over the live suffix still agree
    tr.head = 60
    while length(tr.rows) < tr.cap   # force the rebuild path
        t += 1
        row = (time = t, x = 1)
        push!(rows, row)
        CausalFrames.treepush!(tr, stateprotos, row)
    end
    t += 1
    row = (time = t, x = 2)
    push!(rows, row)
    CausalFrames.treepush!(tr, stateprotos, row)   # triggers rebuild!
    dropped = length(rows) - length(tr.rows)
    @test tr.head == 1 && dropped == 59
    @test queryvals(tr, 1, length(tr.rows)) == naive(rows[(dropped+1):end])

    # windowstart uses the kernel's exact membership predicate
    times = [1, 2, 2, 3, 5]
    @test CausalFrames.windowstart(times, 1, 5, 3) == 2
    @test CausalFrames.windowstart(times, 1, 2, 0) == 2   # ties at the edge
    @test CausalFrames.windowstart(times, 1, 5, 100) == 1  # everything in
    @test CausalFrames.windowstart(times, 1, 100, 1) == 6  # nothing in
    @test CausalFrames.windowstart(times, 4, 5, 3) == 4    # head clips
    ftimes = [0.1, 0.2, 0.3]
    @test CausalFrames.windowstart(ftimes, 1, 0.3, 0.3 - 0.2) ==
          findfirst(s -> 0.3 - s <= 0.3 - 0.2, ftimes)
end
