# Precompile the main pipeline paths so first use is fast. User-supplied row
# functions and summarizer combinations still specialize at first call — this
# covers the shared machinery (chunk iteration, CSV reading, frame assembly,
# the folding kernels for the stock summarizers over Int and Float64 columns).
@setup_workload begin
    dir = mktempdir()
    csv = joinpath(dir, "precompile.csv")
    write(csv, "time,sym,qty\n1,a,1.0\n2,b,2.0\n2,a,3.0\n3,a,4.0\n")
    @compile_workload begin
        ctx = Context(0, 10)
        p = readcsv(csv) |>
            filterrows(r -> r.qty > 0) |>
            addcolumns(r -> (; v = 2.0 * r.qty))
        DataFrame(load(ctx, p |> summarize([Count(), Sum(:v), SumPower(:v, 2),
                                            Moment(:v, 2), Min(:v), Max(:v),
                                            Product(:v), Mean(:v), Variance(:v),
                                            Std(:v), DotProduct(:v, :qty),
                                            Covariance(:v, :qty),
                                            Correlation(:v, :qty)])))
        DataFrame(load(ctx, p |> summarize([Count(), Sum(:v)]; key = :sym)))
        DataFrame(load(ctx, p |> summarizecycles(Sum(:v); key = :sym)))
        DataFrame(load(ctx, p |> addsummarycolumns([First(:v), Last(:v)])))
        DataFrame(load(ctx, p |> asofjoin(readcsv(csv); key = :sym,
                                          rightprefix = "r", righttime = :rt)))
        DataFrame(load(ctx, clock(1) |> asofjoin(readcsv(csv); tolerance = 2,
                                                 rightprefix = "r")))
        foreach(DataFrame, stream(ctx, clock(1) |> summarize(Count())))
        DataFrame(load(ctx, emptyframe()))
    end
end
