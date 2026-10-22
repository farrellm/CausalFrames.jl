# Precompile the main pipeline paths so first use is fast: the shared machinery
# (chunk iteration, sources, frame assembly) and the stock summarizers' kernels
# over Int and Float64 columns. User row functions and summarizer combinations
# still specialize at first call.
@setup_workload begin
    dir = mktempdir()
    csv = joinpath(dir, "precompile.csv")
    write(csv, "time,sym,qty\n1,a,1.0\n2,b,2.0\n2,a,3.0\n3,a,4.0\n")
    csvtypes = Dict(:time => Int, :qty => Float64)
    @compile_workload begin
        ctx = Context(0, 10)
        ld(q) = DataFrame(load(ctx, q))
        p =
            readcsv(csv; types = csvtypes) |>
            filterrows(r -> r.qty > 0) |>
            addcolumns(r -> (; v = 2.0 * r.qty))
        ld(
            p |> summarize([Count(), Sum(:v), SumPower(:v, 2),
                Moment(:v, 2), Min(:v), Max(:v), MaxIndex(:v),
                Product(:v), Mean(:v), Variance(:v),
                Std(:v), DotProduct(:v, :qty),
                Covariance(:v, :qty),
                Correlation(:v, :qty)]),
        )
        ld(readcsv(csv; types = csvtypes, sort = true))
        # The regressions get their own workload: each expands to a dependency
        # per cross product, and a prototype tuple much past 32 falls off
        # inference. Both solve paths run (the K = 1 closed form and the
        # factorization), over non-collinear predictors so the Cholesky
        # succeeds and the coefficient and standard-error code runs.
        ld(
            p |> addcolumns(r -> (; u = r.qty * r.qty)) |>
            summarize([LinearRegression(:qty, :v; name = :m1),
                LinearRegression([:qty, :u], :v; name = :m2)]),
        )
        ld(p |> selectcolumns(:sym, r"^q") |> dropcolumns(:sym))
        ld(p |> reordercolumns(:v, r"^q"))
        ld(p |> lag(1))
        ld(p |> head(2))
        ld(p |> settime(r -> r.time + 1))
        # the Symbol form's rename branch, not the `settime(:time)` early return
        ld(p |> addcolumns(r -> (; t2 = r.time + 1)) |> settime(:t2))
        ld(p |> lastrow())
        ld(p |> lastrow(; key = :sym))
        # the fixture's cycle at time 2 arrives as (b, a), so both forms reorder
        ld(p |> sortcycles(:sym))
        ld(p |> sortcycles(r -> r.qty; rev = true))
        # the fills need a Missing-admitting column, which nothing else here has
        pm = p |> addcolumns(r -> (; m = isodd(r.time) ? missing : r.qty))
        ld(pm |> forwardfill(:m))
        ld(pm |> forwardfill(:m; key = :sym, tolerance = 2))
        ld(pm |> fillmissing(:m => 0.0))
        DataFrame(load(Context(2, 5), pm |> warmup(1, forwardfill(:m))))
        ld(concatenate(p |> filterrows(r -> r.time <= 2),
            p |> filterrows(r -> r.time > 2)))
        ld(
            merge(p |> filterrows(r -> r.time <= 2) |> dropcolumns(:sym),
                p |> filterrows(r -> r.time > 1)),
        )
        ld(p |> summarize([Count(), Sum(:v)]; key = :sym))
        ld(p |> summarizecycles(Sum(:v); key = :sym))
        # A row term: the term-column setup (the kernels are the ones above).
        ld(p |> summarize(Sum(:vq => r -> r.v * r.qty)))
        ld(p |> intervalize(clock(2), [Count(), Sum(:v),
            Mean(:v)]; closelast = true))
        ld(p |> intervalize(clock(2), [Count(), Sum(:v)];
            key = :sym))
        # the running window tier keyless; then keyed, which also takes the
        # vanish-row path, with windowed trackers in the running tier beside
        # the monoid tier (Product is only a monoid) and a fieldless dependent
        ld(p |> summarizewindows(clock(2), 3,
            [Count(), Sum(:v), Mean(:v)]))
        ld(
            p |> summarizewindows(clock(2), 3,
                [Min(:v), Last(:v), Product(:v), Mean(:v)]; key = :sym),
        )
        # the declared-key (dense) paths: the slot fold shared by cycles and
        # intervals, and the running window tier's dense emission
        ld(p |> summarizecycles(Sum(:v); key = :sym, keyset = ["a", "b"]))
        ld(p |> intervalize(clock(2), [Count(), Sum(:v)]; key = :sym,
            keyset = ["a", "b"]))
        ld(
            p |> summarizewindows(clock(2), 3, [Count(), Sum(:v)];
                key = :sym, keyset = ["a", "b"]),
        )
        ld(p |> addsummarycolumns([First(:v), Last(:v)]))
        # group summarizers take the running tier and other monoids the tree
        # tier; only user summarizers without that structure reach the refold
        # tier, which specializes at first call
        ld(
            p |> addrollingcolumns((w2 = 2,),
                [Sum(:v), Mean(:v), AgeWeightedSum(:v)];
                key = :sym),
        )
        # a Bars window beside a time window: per-key rings plus the buffer
        ld(p |> addrollingcolumns((b3 = Bars(3), w2 = 2),
            [Sum(:v), Mean(:v)]; key = :sym))
        ld(
            clock(1) |> addrollingcolumns(
                (w1 = 1, w3 = 3), [Count(), Min(:qty)];
                from = readcsv(csv; types = csvtypes)),
        )
        ld(
            p |> addrollingcolumns((w2 = 2,),
                [Min(:v), Last(:v), Product(:v), Mean(:v), MaxWithIndex(:v),
                    Quantile(:v, [0.25, 0.75]), Median(:v), PercentRank(:v),
                    MeanAbsDev(:v)];
                key = :sym),
        )
        ld(
            p |> asofjoin(readcsv(csv; types = csvtypes); key = :sym,
                rightprefix = "r", righttime = :rt),
        )
        ld(
            clock(1) |> asofjoin(readcsv(csv; types = csvtypes); tolerance = 2,
                rightprefix = "r"),
        )
        ld(
            p |> Acausal.futurejoin(readcsv(csv; types = csvtypes);
                key = :sym, rightprefix = "r", righttime = :rt),
        )
        # a timeless table with a String value column, in the default mode and
        # the row-dropping one
        dim = (sym = ["a"], sector = ["tech"])
        ld(p |> lookupjoin(dim; key = :sym))
        ld(p |> lookupjoin(dim; key = :sym, unmatched = :drop))
        ld(p |> Acausal.lead(1))
        ld(p |> Acausal.settime(r -> r.time - 1))
        foreach(DataFrame, stream(ctx, clock(1) |> summarize(Count())))
        scan(ctx, p |> writecsv(joinpath(dir, "precompile-out.csv")))
        jls = joinpath(dir, "precompile.jls")
        scan(ctx, p |> writejls(jls))
        ld(readjls(jls))
        # readtable's three paths: a DataFrame plain and renamed-and-sorted, the
        # generic path over a NamedTuple of vectors, and a frame read back
        tdf = DataFrame(ts = [3, 1, 2], x = [1.0, 2.0, 3.0])
        ld(readtable(tdf; time = :ts, sort = true))
        ld(readtable(DataFrame(time = [1, 2, 3], x = [1.0, 2.0, 3.0])))
        ld(readtable((time = [1, 2, 3], x = [1.0, 2.0, 3.0])))
        ld(readtable(load(ctx, p)))
        ld(emptyframe())
    end
end
