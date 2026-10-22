@testset "monoid queues" begin
    intypes = (time = Int64, x = Int64)
    R = @NamedTuple{time::Int64, x::Int64}
    # a state tuple mixing an invertible accumulator, order-insensitive and
    # order-sensitive trackers, and the monoid-only product
    ss = (Sum(:x), Min(:x), Last(:x), Product(:x))
    stateprotos = map(s -> CausalFrames.fresh(s, intypes), ss)
    names = Val((:x_sum, :x_min, :x_last, :x_product))
    Q = CausalFrames.queuestype(stateprotos, R, Int64)

    naive(rows) = begin
        sts = map(CausalFrames.fresh, stateprotos)
        foreach(r -> foreach(st -> CausalFrames.update!(st, r), sts), rows)
        CausalFrames.summaryvalues(sts, names)
    end
    # the states are borrowed scratch, valid only until the next query, so they
    # are read straight through summaryvalues, as the window kernels do
    queryvals(mq, w) = begin
        sts = CausalFrames.queuestates(mq, w)
        sts === nothing ? nothing : CausalFrames.summaryvalues(sts, names)
    end
    # a window's rows at t, by the kernels' membership test, or (Bars) the
    # newest n, nothing until n have arrived
    inwindow(rows, t, lb) = filter(r -> t - r.time <= lb, rows)
    inwindow(rows, t, lb::Bars) =
        length(rows) < lb.n ? nothing : rows[(end-lb.n+1):end]
    expected(rows, t, lb) = begin
        live = inwindow(rows, t, lb)
        live === nothing || isempty(live) ? nothing : naive(live)
    end

    # Pseudo-random pushes with deliberate ties (zero steps) and gaps longer
    # than the short window, so a window empties and refills; queries at the
    # newest row's time or later, skipped on some rows so a back accumulates
    # rows that expire before any flip folds them. Products of ±1/±2 stay
    # exact, so any association agrees with the naive fold.
    lookbacks = (3, 25, Bars(4))
    steps = lcgsequence(7, 600, 3) .+ 4 .* (lcgsequence(5, 600, 40) .== 0)
    xs = map(v -> (-2, -1, 1, 2)[v+1], lcgsequence(11, 600, 4))
    skips = lcgsequence(13, 600, 3)
    mq = CausalFrames.newqueues(Q, stateprotos, length(lookbacks))
    rows = R[]
    t = 0
    bad = Tuple{Int,Int}[]   # (row, window) of every query that disagreed
    for i in 1:600
        t += steps[i]
        row = (time = t, x = xs[i])
        push!(rows, row)
        CausalFrames.queuepush!(mq, row, lookbacks)
        skips[i] == 0 && continue
        # sometimes after the newest row, but never after the next: rows
        # arrive at or after every query's time
        τ = t + (skips[i] == 2 && i < 600 && steps[i+1] > 0)
        for (w, lb) in enumerate(lookbacks)
            lb isa Bars ||
                CausalFrames.expirewindow!(mq, mq.queues[w], τ, lb)
            got =
                lb isa Bars && CausalFrames.nstored(mq) < lb.n ? nothing :
                queryvals(mq, w)
            isequal(got, expected(rows, τ, lb)) || push!(bad, (i, w))
        end
    end
    @test isempty(bad)
    # the dead prefix was dropped, and every window's live rows are still there
    @test mq.dropped > 0
    @test CausalFrames.liverows(mq) ==
          rows[(minimum(q->q.lo, mq.queues)):end]
    @test @inferred(Union{Nothing,Tuple}, CausalFrames.queuestates(mq, 2)) isa Tuple

    # Pushes alone expire each window against the pushed row's time, so a key
    # that is never queried keeps only its windows' rows.
    quiet = CausalFrames.newqueues(Q, stateprotos, 1)
    for t in 1:1000
        CausalFrames.queuepush!(quiet, (time = t, x = 1), (10,))
    end
    @test length(quiet.rows) <= 2 * 64 + 11
    @test queryvals(quiet, 1) == naive([(time = t, x = 1) for t in 990:1000])

    # A steady window allocates nothing once its front has grown: flips reuse
    # the front's state tuples. Asserted as a slope rather than
    # `@allocated(one call) == 0`, since a lone `@allocated` puts a function
    # boundary around the call, where Julia 1.10 materializes the returned
    # tuple that it elides once the query is inlined into its caller. The
    # window always holds rows here, so it is read through `livestates`, as
    # summarizewindows reads it: on Julia 1.10, passing `queuestates`'
    # `Union{Nothing,…}` straight to `summaryvalues` allocates 48 bytes per
    # query (the kernels test for `nothing` first, or use `livestates`). When
    # 1.10 support is dropped, `queuestates` may be read here directly.
    function steady(mq, t0, n)
        s = 0
        for t in t0:(t0+n-1)
            CausalFrames.queuepush!(mq, (time = t, x = 1), (10,))
            CausalFrames.expirewindow!(mq, mq.queues[1], t, 10)
            s += CausalFrames.summaryvalues(
                CausalFrames.livestates(mq, mq.queues[1]), names).x_sum
        end
        return s
    end
    steady(quiet, 1001, 200)
    base = @allocated steady(quiet, 1201, 100)
    @test (@allocated steady(quiet, 1301, 1000)) <= base + 100

    # expiry uses the kernels' exact membership predicate, including ties at
    # the edge and float times where a rearranged test could differ in the
    # last ulp
    edge = CausalFrames.newqueues(Q, stateprotos, 1)
    for (t, x) in ((1, 1), (2, 1), (2, 2), (3, 1), (5, 2))
        CausalFrames.queuepush!(edge, (time = t, x = x), (100,))
    end
    for (τ, lb, lo) in ((5, 100, 1), (5, 3, 2), (5, 2, 4), (6, 0, 6))
        CausalFrames.expirewindow!(edge, edge.queues[1], τ, lb)
        @test edge.queues[1].lo == lo
    end
    @test CausalFrames.queuestates(edge, 1) === nothing
    FR = @NamedTuple{time::Float64, x::Int64}
    fstates = map(s -> CausalFrames.fresh(s, (time = Float64, x = Int64)), ss)
    fq = CausalFrames.newqueues(CausalFrames.queuestype(fstates, FR, Float64),
        fstates, 1)
    ftimes = [0.1, 0.2, 0.3]
    for ft in ftimes
        CausalFrames.queuepush!(fq, (time = ft, x = 1), (1.0,))
    end
    CausalFrames.expirewindow!(fq, fq.queues[1], 0.3, 0.3 - 0.2)
    @test fq.queues[1].lo == findfirst(s -> 0.3 - s <= 0.3 - 0.2, ftimes)
end
