@testset "Context" begin
    ctx = Context(1, 10)
    @test ctx.start == 1
    @test ctx.stop == 10
    @test timetype(ctx) == Int
    @test timetype(Context(1, 10.0)) == Float64
    @test timetype(Context(DateTime(2026), DateTime(2027))) == DateTime
    @test_throws ArgumentError Context(10, 1)
end
