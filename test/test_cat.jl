function test_cat()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)
  @testset "Concatenation" begin
    A   = sprandn(100, 100, .5)
    B   = randn(100, 10) + 1im * randn(100, 10)
    rd(x) = round(Int64, x)
    # C   = rd.(randn(100, 90) * 30)
    C   = randn(100, 90) * 30
    D   = [A B C]
    Ao  = LinearOperator(A)
    Bo  = LinearOperator(B)
    Co  = LinearOperator(C)
    Do  = [Ao Bo Co]
    Do2 = LinearOperator(D)
    rhs = randn(200)

    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))  # throws InexactError()
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))

    @test_throws LinearOperatorException [LinearOperator(rand(5,5)) opEye(3)]

    # test vcat
    A   = sprandn(100, 100, 0.5)
    B   = randn(10, 100) + 1im * randn(10, 100)
    # C   = rd.(randn(90, 100) * 30)
    C   = randn(90, 100) * 30
    D   = [A; B; C]
    Ao  = LinearOperator(A)
    Bo  = LinearOperator(B)
    Co  = LinearOperator(C)
    Do  = [Ao; Bo; Co]
    Do2 = LinearOperator(D)
    rhs = randn(100)

    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))

    @test_throws LinearOperatorException [LinearOperator(rand(5,5)) ; opEye(3)]
    K = [Matrix(1.0I, 2, 2) opZeros(2,3) ; opZeros(3,2) opEye(3)]
    v = rand(5)
    @test all(v .== K * v)

    K = [opEye(2) ; sparse(1.0I, 2, 2)]
    v = rand(2)
    @test all(K * v .== [v ; v])
  end
end

test_cat()
