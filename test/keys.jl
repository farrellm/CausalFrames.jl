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
        df = DataFrame(load(Context(1, 7),
            intervalize(kp, clock(3), Sum(:x); key = :k)))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x_sum == [1.0, 3.0, 2.0]

        df = loadk(addsummarycolumns(Sum(:x); key = :k))
        @test df.x_sum == [1.0, 2.0, 3.0, 6.0, 6.0, 9.0]

        df = loadk(lastrow(; key = :k))
        @test isequal(df.k, ["a", "b", missing])
        @test df.x == [5.0, 6.0, 4.0]
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
        # missing. Up to three key columns: from four, Base's NamedTuple
        # `hash` allocates on any `Union` field (issue #116). Key `j` of row
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
            load(Context(0, n + 1), p |> t(keys, src))
            return @allocations load(Context(0, n + 1), p |> t(keys, src))
        end
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
        ]
        # On Julia 1.10, Base's NamedTuple `hash` allocates from two `Union`
        # fields (issue #116). When 1.10 support is dropped, test every key
        # set on every version.
        keysets = VERSION < v"1.11" ? ([:k1],) : ([:k1], [:k1, :k2], [:k1, :k2, :k3])
        for t in ops, keys in keysets
            @test keyallocs(t, keys, 8000) - keyallocs(t, keys, 2000) < 200
        end
    end
end
