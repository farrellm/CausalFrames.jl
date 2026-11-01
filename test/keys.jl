@testset "key columns admitting missing" begin
    # Keys take the key columns' declared types (issue #114): a key column
    # typed `Union{Missing,String}` gives `Union`-typed keys, whatever a given
    # row holds, so lookups never convert a narrower key.
    kdf = DataFrame(time = 1:6,
        k = Union{Missing,String}["a", missing, "b", missing, "a", "b"],
        x = 1.0:6.0)
    kp = readtable(kdf)
    loadk(t) = DataFrame(load(Context(1, 7), kp |> t))

    @testset "a missing key is a group of its own" begin
        # The group table is keyed by the column's type, not by the first
        # row's value (which typed it `String`, and a missing key then threw).
        df = loadk(summarize(Sum(:x); key = :k))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x_sum == [6.0, 9.0, 6.0]
        @test eltype(df.k) == Union{Missing,String}

        df = loadk(summarizecycles(Sum(:x); key = :k))
        @test isequal(df.k, kdf.k)
        @test df.x_sum == kdf.x

        # (the one closed interval is [1, 4), timestamped 4)
        df = loadk(intervalize(clock(3), Sum(:x); key = :k))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x_sum == [1.0, 3.0, 2.0]

        df = loadk(addsummarycolumns(Sum(:x); key = :k))
        @test df.x_sum == [1.0, 2.0, 3.0, 6.0, 6.0, 9.0]

        df = loadk(lastrow(; key = :k))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x == [5.0, 6.0, 4.0]
    end

    @testset "keys order field by field, missing last" begin
        # `keyisless` must agree with `isless` on the keys' value tuples,
        # which the sorts in `closecycle!` and `sortedgroups` rely on.
        df2 = DataFrame(time = fill(1, 6),
            a = Union{Missing,String}["b", missing, "a", "b", missing, "a"],
            b = Union{Missing,Int}[2, 1, missing, missing, missing, 1],
            x = 1.0:6.0)
        p2 = readtable(df2)
        want = sort!(collect(zip(df2.a, df2.b)))
        for t in (summarize(Sum(:x); key = [:a, :b]),
            summarizecycles(Sum(:x); key = [:a, :b]))
            out = DataFrame(load(Context(1, 2), p2 |> t))
            @test isequal(collect(zip(out.a, out.b)), want)
        end
        ks = [(a = missing, b = 1), (a = "a", b = missing), (a = "a", b = 1),
            (a = missing, b = missing), (a = "b", b = 2)]
        for x in ks, y in ks
            @test CausalFrames.keyisless(x, y) == isless(values(x), values(y))
        end
    end

    @testset "a key column widening across chunks" begin
        # The first chunk's key column is `String`, the second's admits
        # `missing`: the group table re-keys to the wider type.
        c1 = DataFrame(time = [1, 2], k = ["a", "b"], x = [1.0, 2.0])
        c2 = DataFrame(time = [3, 4], k = Union{Missing,String}[missing, "a"],
            x = [3.0, 4.0])
        p = CausalPipeline(ctx -> [c1, c2])
        df = DataFrame(load(Context(1, 5), p |> summarize(Sum(:x); key = :k)))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x_sum == [5.0, 2.0, 3.0]
        df = DataFrame(load(Context(1, 5),
            p |> summarizecycles(Sum(:x); key = :k)))
        @test isequal(df.k, [c1.k; c2.k])
    end

    @testset "allocations do not grow with rows" begin
        # Per-row allocations, as the slope between two run lengths: once a
        # key column admitted `missing`, every row's key was built narrower
        # than the store's and converted on lookup, even with no value
        # missing. From four key columns (two on Julia 1.10), Base's
        # NamedTuple `hash` allocated on any `Union` field too, until the
        # stores hashed through `KeyDict` (issue #116). Key `j` of row
        # `i` is `(i * j) % 5`, so the first five rows hold every key
        # combination: the lookup table.
        keycol(j, n) = Vector{Union{Missing,String}}(
            ["s" * string((i * j) % 5) for i in 1:n])
        function keyed!(df, keys)
            for (j, c) in enumerate(keys)
                df[!, c] = keycol(j, nrow(df))
            end
            return df
        end
        function keyallocs(t, keys, n)
            src = keyed!(DataFrame(time = 1:n, x = Float64.(mod.(1:n, 7))), keys)
            p = CausalPipeline(ctx -> [src])
            return @allocations load(Context(0, n + 1), p |> t(keys, src))
        end
        # `s`, the source, is for `asofjoin`'s right side
        ops = [
            (k, s) -> lastrow(; key = k),
            (k, s) -> forwardfill(:x; key = k),
            (k, s) -> summarize(Sum(:x); key = k),
            (k, s) -> summarizecycles(Sum(:x); key = k),
            (k, s) -> addsummarycolumns(Sum(:x); key = k),
            (k, s) -> addrollingcolumns((w5 = 5, b5 = Bars(5)), Sum(:x); key = k),
            (k, s) -> intervalize(clock(10), Sum(:x); key = k),
            (k, s) -> summarizewindows(clock(10), 20, Sum(:x); key = k),
            (k, s) -> asofjoin(CausalPipeline(ctx -> [s]) |> selectcolumns(k..., :x);
                key = k, rightprefix = "r_"),
            (k, s) -> lookupjoin(keyed!(DataFrame(v = fill(1.0, 5)), k); key = k),
            (k, s) -> CausalFrames.Acausal.futurejoin(
                CausalPipeline(ctx -> [s]) |> selectcolumns(k..., :x);
                key = k, rightprefix = "r_"),
        ]
        keysets = ([:k1], [:k1, :k2, :k3, :k4], [Symbol(:k, j) for j in 1:6])
        for t in ops, keys in keysets
            keyallocs(t, keys, 10) # compile
            @test keyallocs(t, keys, 8000) - keyallocs(t, keys, 2000) < 200
        end
    end

    @testset "KeyDict" begin
        # Stored keys hash and compare field by field, so a lookup key of
        # narrower or numerically different types finds its entry unconverted.
        K = @NamedTuple{a::Union{Missing,String}, b::Float64}
        d = CausalFrames.KeyDict{K,Int}()
        d[(a = "x", b = 1.0)] = 1
        d[(a = missing, b = 2.0)] = 2
        @test d[(a = "x", b = 1)] == 1
        @test get(d, (a = missing, b = 2), 0) == 2
        @test get(d, (a = "x", b = 2), 0) == 0
        @test haskey(d, (a = missing, b = 2.0))
        @test get!(d, (a = "y", b = 3), 3) == 3
        @test length(d) == 3
        @test eltype(collect(d)) == Pair{K,Int}
        @test isequal(first.(CausalFrames.sortedgroups(d)),
            K[(a = "x", b = 1.0), (a = "y", b = 3.0), (a = missing, b = 2.0)])
        filter!(p -> last(p) != 3, d)
        @test sort!(collect(values(d))) == [1, 2]
        delete!(d, (a = "x", b = 1))
        @test isequal(collect(d), [(a = missing, b = 2.0) => 2])
        @test isempty(empty!(d))
    end
end
