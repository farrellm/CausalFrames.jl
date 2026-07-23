# Precompile the main pipeline paths so first use is fast. User-supplied row
# functions and summarizer combinations still specialize at first call — this
# covers the shared machinery (chunk iteration, CSV reading, frame assembly,
# the folding kernels for the stock summarizers over Int and Float64 columns).
@setup_workload begin
    dir = mktempdir()
    csv = joinpath(dir, "precompile.csv")
    write(csv, "time,sym,qty\n1,a,1.0\n2,b,2.0\n2,a,3.0\n3,a,4.0\n")
    csvtypes = Dict(:time => Int, :qty => Float64)
    @compile_workload begin
        ctx = Context(0, 10)
        p =
            readcsv(csv; types = csvtypes) |>
            filterrows(r -> r.qty > 0) |>
            addcolumns(r -> (; v = 2.0 * r.qty))
        DataFrame(
            load(
                ctx,
                p |> summarize([Count(), Sum(:v), SumPower(:v, 2),
                    Moment(:v, 2), Min(:v), Max(:v),
                    Product(:v), Mean(:v), Variance(:v),
                    Std(:v), DotProduct(:v, :qty),
                    Covariance(:v, :qty),
                    Correlation(:v, :qty)]),
            ),
        )
        DataFrame(load(ctx, p |> selectcolumns(:sym, r"^q") |> dropcolumns(:sym)))
        DataFrame(load(ctx, p |> summarize([Count(), Sum(:v)]; key = :sym)))
        DataFrame(load(ctx, p |> summarizecycles(Sum(:v); key = :sym)))
        DataFrame(
            load(
                ctx,
                p |> intervalize(clock(2), [Count(), Sum(:v),
                        Mean(:v)]; closelast = true),
            ),
        )
        DataFrame(load(ctx, p |> intervalize(clock(2), [Count(), Sum(:v)];
            key = :sym)))
        DataFrame(load(ctx, p |> addsummarycolumns([First(:v), Last(:v)])))
        # all-group summarizers take the running window mode, the mixed
        # group/monoid set the tree mode; the re-fold mode is reachable
        # only through user summarizers without the structure, so it
        # specializes at first call like any custom summarizer
        DataFrame(
            load(ctx, p |> addrollingcolumns((w2 = 2,),
                [Sum(:v), Mean(:v)];
                key = :sym)),
        )
        DataFrame(
            load(
                ctx,
                clock(1) |> addrollingcolumns(
                    (w1 = 1, w3 = 3), [Count(), Min(:qty)];
                    from = readcsv(csv; types = csvtypes)),
            ),
        )
        DataFrame(
            load(ctx, p |> addrollingcolumns((w2 = 2,),
                [Min(:v), Last(:v)];
                key = :sym)),
        )
        DataFrame(
            load(
                ctx,
                p |> asofjoin(readcsv(csv; types = csvtypes); key = :sym,
                    rightprefix = "r", righttime = :rt),
            ),
        )
        DataFrame(
            load(
                ctx,
                clock(1) |> asofjoin(readcsv(csv; types = csvtypes); tolerance = 2,
                    rightprefix = "r"),
            ),
        )
        DataFrame(
            load(
                ctx,
                p |> Acausal.futurejoin(readcsv(csv; types = csvtypes);
                    key = :sym, rightprefix = "r", righttime = :rt),
            ),
        )
        foreach(DataFrame, stream(ctx, clock(1) |> summarize(Count())))
        scan(ctx, p |> writecsv(joinpath(dir, "precompile-out.csv")))
        DataFrame(load(ctx, emptyframe()))
    end
end
