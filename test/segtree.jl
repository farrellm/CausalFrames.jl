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
    # the query result is borrowed scratch, valid only until the next query,
    # so it is read straight through summaryvalues, as the window kernels do
    queryvals(tr, lo, hi) =
        CausalFrames.summaryvalues(
            CausalFrames.treequery(tr, lo, hi),
            Val((:x_sum, :x_min, :x_last, :x_product)))

    # pseudo-random pushes cross-checked against a naive re-fold, across
    # several capacity-doubling rebuilds; products of ±1/±2 stay exact
    steps = lcgsequence(7, 100, 3)      # deliberate ties (zero steps)
    xs = map(v -> (-2, -1, 1, 2)[v+1], lcgsequence(11, 100, 4))
    picks = lcgsequence(13, 200, 2^30)
    tr = CausalFrames.newsegtree(stateprotos, R, Int64)
    rows = R[]
    t = 0
    # one assertion per walk, naming every (lo, hi) query that disagreed
    bad = Tuple{Int,Int}[]
    for i in 1:100
        t += steps[i]
        row = (time = t, x = xs[i])
        push!(rows, row)
        CausalFrames.treepush!(tr, stateprotos, row)
        lo = 1 + picks[2i-1] % i
        hi = lo + picks[2i] % (i - lo + 1)
        for (a, b) in ((lo, hi), (i, i), (1, i))
            queryvals(tr, a, b) == naive(rows[a:b]) || push!(bad, (a, b))
        end
    end
    @test isempty(bad)
    @test @inferred(CausalFrames.treequery(tr, 1, length(rows))) isa Tuple

    # The query accumulators are the tree's own scratch, so queries don't
    # allocate. Asserted as a slope rather than `@allocated(one call) == 0`: a
    # lone `@allocated` puts a function boundary around the call, where Julia
    # 1.10 materializes the returned tuple (32-48 bytes) that it elides once the
    # query is inlined into its caller.
    querytotal(tr, n) = sum(_ -> queryvals(tr, 1, length(tr.rows)).x_sum, 1:n)
    querytotal(tr, 2)
    base = @allocated querytotal(tr, 100)
    @test (@allocated querytotal(tr, 1000)) <= base + 100

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

    # summarizewindows' batched protocol: runs of bare appends, one sync, then
    # queries. The head stays put at first, so capacity grows through
    # reallocating rebuilds; then it slides as a window would, keeping few live
    # rows, so rebuilds recur at an unchanged capacity (the swapping path). Odd
    # lengths put the last parent's right child past the end, and a head slid
    # past the first unsynced leaf starts the sync at the head.
    runs = lcgsequence(17, 80, 9)
    ahead = lcgsequence(19, 80, 5)
    tr = CausalFrames.newsegtree(stateprotos, R, Int64)
    rows = R[]
    swapped = 0
    bad = Tuple{Int,Int,Int}[]
    for step in 1:80
        before, nodes0 = length(tr.rows), tr.nodes
        k = runs[step] + 1
        for _ in 1:k
            t += 1
            row = (time = t, x = xs[1+t%100])
            push!(rows, row)
            CausalFrames.treeappend!(tr, stateprotos, row)
        end
        length(tr.rows) < before + k && (swapped += tr.nodes === nodes0)
        n = length(tr.rows)
        step > 20 && (tr.head = max(tr.head, n - ahead[step]))
        CausalFrames.treesync!(tr)
        off = length(rows) - n
        for lo in tr.head:n
            queryvals(tr, lo, n) == naive(rows[(off+lo):end]) ||
                push!(bad, (step, lo, n))
            queryvals(tr, tr.head, lo) == naive(rows[(off+tr.head):(off+lo)]) ||
                push!(bad, (step, tr.head, lo))
        end
    end
    @test isempty(bad)
    @test swapped > 0

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
