@testset "lookupjoin" begin
    onechunk(; cols...) = CausalPipeline(ctx -> [DataFrame(; cols...)])
    ctx = Context(0, 10)
    loaddf(p) = DataFrame(load(ctx, p))

    @testset "basic" begin
        left = onechunk(time = [1, 2, 3, 4], sym = ["a", "b", "c", "a"],
            px = [1.0, 2.0, 3.0, 4.0])
        dim = DataFrame(sector = ["tech", "energy"], sym = ["a", "b"],
            lot = [100, 10])
        df = loaddf(left |> lookupjoin(dim; key = :sym))
        # input columns first, then the table's value columns in its own order
        @test names(df) == ["time", "sym", "px", "sector", "lot"]
        @test df.time == [1, 2, 3, 4]
        @test df.px == [1.0, 2.0, 3.0, 4.0]
        @test isequal(df.sector, ["tech", "energy", missing, "tech"])
        @test isequal(df.lot, [100, 10, missing, 100])
        @test eltype(df.sector) == Union{Missing,String}
        @test eltype(df.lot) == Union{Missing,Int}

        # any Tables.jl table: a NamedTuple of vectors, a vector of NamedTuples
        for tab in ((sym = ["a", "b"], lot = [100, 10]),
            [(sym = "a", lot = 100), (sym = "b", lot = 10)])
            df = loaddf(left |> lookupjoin(tab; key = :sym))
            @test isequal(df.lot, [100, 10, missing, 100])
        end

        # multi-key: every key column must match
        left2 = onechunk(time = [1, 1, 2], sym = ["a", "a", "b"],
            venue = ["x", "y", "x"])
        tab = (venue = ["y", "x"], sym = ["a", "b"], fee = [0.5, 0.25])
        df = loaddf(left2 |> lookupjoin(tab; key = [:sym, :venue]))
        @test names(df) == ["time", "sym", "venue", "fee"]
        @test isequal(df.fee, [missing, 0.5, 0.25])

        # a table of keys alone appends nothing; with :drop it is a semi-join
        keysonly = (sym = ["a"],)
        df = loaddf(
            onechunk(time = [1, 2], sym = ["a", "z"]) |>
            lookupjoin(keysonly; key = :sym, unmatched = :drop),
        )
        @test names(df) == ["time", "sym"]
        @test df.sym == ["a"]
    end

    @testset "unmatched" begin
        left = onechunk(time = [1, 2, 3], sym = ["a", "z", "b"])
        dim = (sym = ["a", "b"], lot = [100, 10])

        df = loaddf(left |> lookupjoin(dim; key = :sym, unmatched = :drop))
        @test df.time == [1, 3]
        @test df.sym == ["a", "b"]
        @test df.lot == [100, 10]
        @test eltype(df.lot) == Int

        err = @test_throws ArgumentError loaddf(
            left |>
            lookupjoin(dim; key = :sym, unmatched = :error),
        )
        @test occursin("\"z\"", sprint(showerror, err.value))

        # every key present: both strict modes keep the table's element type
        matched = onechunk(time = [1, 2], sym = ["b", "a"])
        for unmatched in (:error, :drop)
            df = loaddf(matched |> lookupjoin(dim; key = :sym, unmatched))
            @test df.lot == [10, 100]
            @test eltype(df.lot) == Int
        end

        # a wholly unmatched chunk is skipped under :drop, the next still emitted
        chunks = CausalPipeline(
            ctx -> [DataFrame(time = [1, 2], sym = ["y", "z"]),
                DataFrame(time = [3], sym = ["a"])],
        )
        df = loaddf(chunks |> lookupjoin(dim; key = :sym, unmatched = :drop))
        @test df.time == [3]
        @test df.lot == [100]
        frame = load(ctx,
            onechunk(time = [1], sym = ["z"]) |>
            lookupjoin(dim; key = :sym, unmatched = :drop))
        @test nrow(frame) == 0

        @test_throws ArgumentError lookupjoin(dim; key = :sym, unmatched = :inner)
    end

    @testset "construction errors" begin
        @test_throws ArgumentError lookupjoin((sym = ["a", "a"], lot = [1, 2]);
            key = :sym)
        @test_throws ArgumentError lookupjoin(
            (sym = ["a", "a"], v = ["x", "x"], lot = [1, 2]); key = [:sym, :v])
        # distinct multi-keys sharing a component are not duplicates
        lookupjoin((sym = ["a", "a"], v = ["x", "y"], lot = [1, 2]);
            key = [:sym, :v])

        # a timed table, a frame included, belongs to asofjoin
        @test_throws ArgumentError lookupjoin((time = [1], sym = ["a"]); key = :sym)
        frame = load(ctx, onechunk(time = [1], sym = ["a"]))
        @test_throws ArgumentError lookupjoin(frame; key = :sym)

        @test_throws ArgumentError lookupjoin((sym = ["a"],); key = :nope)
        @test_throws ArgumentError lookupjoin((sym = ["a"],))
        @test_throws ArgumentError lookupjoin((sym = ["a"],); key = Symbol[])
        @test_throws ArgumentError lookupjoin((sym = ["a"], v = [1]);
            key = [:sym, :sym])
        @test_throws ArgumentError lookupjoin(42; key = :sym)
        # a prefixed table column colliding with a key
        @test_throws ArgumentError lookupjoin((r_x = ["a"], x = [1]); key = :r_x,
            rightprefix = "r")
    end

    @testset "run-time checks and prefixes" begin
        dim = (sym = ["a"], px = [1.0])
        @test_throws ArgumentError loaddf(
            onechunk(time = [1], other = ["a"]) |>
            lookupjoin(dim; key = :sym),
        )

        left = onechunk(time = [1], sym = ["a"], px = [2.0])
        @test_throws ArgumentError loaddf(left |> lookupjoin(dim; key = :sym))
        df = loaddf(left |>
                    lookupjoin(dim; key = :sym, leftprefix = :l, rightprefix = "r"))
        @test names(df) == ["time", "sym", "l_px", "r_px"]
        @test df.l_px == [2.0]
        @test isequal(df.r_px, [1.0])
        df = loaddf(left |> lookupjoin(dim; key = :sym, rightprefix = "r"))
        @test names(df) == ["time", "sym", "px", "r_px"]

        # a left prefix colliding with a table column
        @test_throws ArgumentError loaddf(
            onechunk(time = [1], sym = ["a"], x = [1]) |>
            lookupjoin((sym = ["a"], l_x = [2]); key = :sym, leftprefix = "l"))
    end

    @testset "key types" begin
        # isequal across numeric types, without conversion
        df = loaddf(
            onechunk(time = [1, 2], id = [1, 2]) |>
            lookupjoin((id = [2.0, 1.0], v = ["two", "one"]); key = :id),
        )
        @test isequal(df.v, ["one", "two"])
        # missing matches missing
        df = loaddf(
            onechunk(time = [1, 2], id = [missing, 1]) |>
            lookupjoin((id = [missing, 1], v = ["m", "one"]); key = :id),
        )
        @test isequal(df.v, ["m", "one"])
    end

    @testset "chunks, streams and split contexts" begin
        dim = (sym = ["a", "b"], lot = [100, 10])
        src = concatenate(
            readtable(DataFrame(time = 0:4, sym = ["a", "b", "c", "a", "b"])),
            readtable(DataFrame(time = 5:9, sym = ["c", "a", "b", "c", "a"])))
        whole = loaddf(src |> lookupjoin(dim; key = :sym))
        @test isequal(whole.lot,
            [100, 10, missing, 100, 10, missing, 100, 10, missing, 100])
        for unmatched in (:missing, :drop)
            joined = src |> lookupjoin(dim; key = :sym, unmatched)
            expected = loaddf(joined)
            streamed = reduce(vcat, [DataFrame(f) for f in stream(ctx, joined)])
            @test isequal(streamed, expected)
            # stateless, so split contexts concatenate too
            split = vcat(DataFrame(load(Context(0, 3), joined)),
                DataFrame(load(Context(3, 10), joined)))
            @test isequal(split, expected)
        end
    end

    @testset "empty table and empty input" begin
        empty = DataFrame(sym = String[], lot = Int[])
        df = loaddf(onechunk(time = [1], sym = ["a"]) |> lookupjoin(empty; key = :sym))
        @test names(df) == ["time", "sym", "lot"]
        @test isequal(df.lot, [missing])
        @test eltype(df.lot) == Union{Missing,Int}

        frame = load(ctx, emptyframe() |> lookupjoin((sym = ["a"], lot = [1]);
            key = :sym))
        @test nrow(frame) == 0
        @test names(frame) == ["time"]
    end

    @testset "table is copied" begin
        dim = DataFrame(sym = ["a"], lot = [1])
        p = onechunk(time = [1], sym = ["a"]) |> lookupjoin(dim; key = :sym)
        dim.lot[1] = 99
        dim.sym[1] = "z"
        @test isequal(loaddf(p).lot, [1])
    end

    @testset "uncurried form" begin
        left = onechunk(time = [1, 2], sym = ["a", "z"])
        dim = (sym = ["a"], lot = [1])
        curried = left |> lookupjoin(dim; key = :sym, unmatched = :drop)
        uncurried = lookupjoin(left, dim; key = :sym, unmatched = :drop)
        @test isequal(loaddf(curried), loaddf(uncurried))
    end

    @testset "kernel allocates nothing per row" begin
        # Only Ints are stored per row, so a String key costs no box; nothing
        # else in the suite would notice if that regressed.
        function lookupalloc()
            n = 200
            syms = ["s" * string(i % 7) for i in 1:n]   # s5, s6 unmatched
            index = Dict((sym = "s" * string(i),) => i + 1 for i in 0:4)
            nt = (time = collect(1:n), sym = syms)
            rows = Vector{Int}(undef, n)
            kn = Val((:sym,))
            call() = CausalFrames.lookuprows!(rows, index, nt, kn)
            call()
            return @allocated call()
        end
        @test lookupalloc() == 0
    end
end
