function test_cat()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)
  @testset ExtendedTestSet "Concatenation" begin
    A = simple_sparse_matrix(ComplexF64, 100, 100)
    B = simple_matrix(ComplexF64, 100, 10)
    C = simple_matrix(ComplexF64, 100, 90)
    D = [A B C]
    Ao = LinearOperator(A)
    Bo = LinearOperator(B)
    Co = LinearOperator(C)
    Do = [Ao Bo Co]
    Do2 = LinearOperator(D)
    rhs = simple_vector(ComplexF64, 200)
    rhs2 = simple_vector(ComplexF64, 100)

    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))  # throws InexactError()
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(transpose(Do) * rhs2 - transpose(D) * rhs2) <= rtol * norm(transpose(D) * rhs2))
    @test(norm(adjoint(Do) * rhs2 - adjoint(D) * rhs2) <= rtol * norm(adjoint(D) * rhs2))

    @test_throws LinearOperatorException [LinearOperator(ones(5, 5)) opEye(3)]

    # test vcat
    A = simple_sparse_matrix(ComplexF64, 100, 100)
    B = simple_matrix(ComplexF64, 10, 100)
    C = simple_matrix(ComplexF64, 90, 100)
    D = [A; B; C]
    Ao = LinearOperator(A)
    Bo = LinearOperator(B)
    Co = LinearOperator(C)
    Do = [Ao; Bo; Co]
    Do2 = LinearOperator(D)
    rhs = simple_vector(ComplexF64, 100)
    rhs2 = simple_vector(ComplexF64, 200)

    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(transpose(Do) * rhs2 - transpose(D) * rhs2) <= rtol * norm(transpose(D) * rhs2))
    @test(norm(adjoint(Do) * rhs2 - adjoint(D) * rhs2) <= rtol * norm(adjoint(D) * rhs2))

    @test_throws LinearOperatorException [LinearOperator(ones(5, 5)); opEye(3)]
    K = [opEye(2) opZeros(2, 3); opZeros(3, 2) opEye(3)]
    v = simple_vector(Float64, 5)
    @test all(v .== K * v)

    K = [opEye(2); sparse(1.0I, 2, 2)]
    v = simple_vector(Float64, 2)
    @test all(K * v .== [v; v])
  end
end

test_cat()
