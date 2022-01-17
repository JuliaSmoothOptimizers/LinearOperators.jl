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
  @testset ExtendedTestSet "Concatenation 3-args" begin
    T = Complex{Float64}
    A = simple_sparse_matrix(T, 100, 100)
    B = simple_matrix(T, 100, 10)
    C = simple_matrix(T, 100, 90)
    D = [A B C]
    Ao = LinearOperator(T, 100, 100, false, false, 
                        (res, v) -> mul!(res, A, v),
                        (res, u) -> mul!(res, transpose(A), u),
                        (res, w) -> mul!(res, adjoint(A), w))
    Bo = LinearOperator(T, size(B)..., false, false, 
                        (res, v) -> mul!(res, B, v),
                        (res, u) -> mul!(res, transpose(B), u),
                        (res, w) -> mul!(res, adjoint(B), w))
    Co = LinearOperator(T, size(C)..., false, false, 
                        (res, v) -> mul!(res, C, v),
                        (res, u) -> mul!(res, transpose(C), u),
                        (res, w) -> mul!(res, adjoint(C), w))
    Do = [Ao Bo Co]
    Do2 = LinearOperator(D)
    rhs = simple_vector(T, 200)
    rhs2 = simple_vector(T, 100)
    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))  # throws InexactError()
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(transpose(Do) * rhs2 - transpose(D) * rhs2) <= rtol * norm(transpose(D) * rhs2))
    @test(norm(adjoint(Do) * rhs2 - adjoint(D) * rhs2) <= rtol * norm(adjoint(D) * rhs2))

    @test_throws LinearOperatorException [LinearOperator(ones(5, 5)) opEye(3)]

    # 5-args mul! hcat
    α, β = 3.0, -4.0
    resD = rand(T, 100)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, D, rhs, α, β)
    mul!(resDo, Do, rhs, α, β)
    mul!(resDo2, Do2, rhs, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))
    resD = rand(T, 200)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, transpose(D), rhs2, α, β)
    mul!(resDo, transpose(Do), rhs2, α, β)
    mul!(resDo2, transpose(Do2), rhs2, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))
    resD = rand(T, 200)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, adjoint(D), rhs2, α, β)
    mul!(resDo, adjoint(Do), rhs2, α, β)
    mul!(resDo2, adjoint(Do2), rhs2, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))

    # test vcat
    A = simple_sparse_matrix(T, 100, 100)
    B = simple_matrix(T, 10, 100)
    C = simple_matrix(T, 90, 100)
    D = [A; B; C]
    Ao = LinearOperator(T, 100, 100, false, false, 
                        (res, v) -> mul!(res, A, v),
                        (res, u) -> mul!(res, transpose(A), u),
                        (res, w) -> mul!(res, adjoint(A), w))
    Bo = LinearOperator(T, size(B)..., false, false, 
                        (res, v) -> mul!(res, B, v),
                        (res, u) -> mul!(res, transpose(B), u),
                        (res, w) -> mul!(res, adjoint(B), w))
    Co = LinearOperator(T, size(C)..., false, false, 
                        (res, v) -> mul!(res, C, v),
                        (res, u) -> mul!(res, transpose(C), u),
                        (res, w) -> mul!(res, adjoint(C), w))
    Do = [Ao; Bo; Co]
    Do2 = LinearOperator(D)
    rhs = simple_vector(T, 100)
    rhs2 = simple_vector(T, 200)

    @test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))
    @test(norm(transpose(Do) * rhs2 - transpose(D) * rhs2) <= rtol * norm(transpose(D) * rhs2))
    @test(norm(adjoint(Do) * rhs2 - adjoint(D) * rhs2) <= rtol * norm(adjoint(D) * rhs2))

    @test_throws LinearOperatorException [LinearOperator(ones(5, 5)); opEye(3)]

    # 5-args mul! hcat
    α, β = 3.0, -4.0
    resD = rand(T, 200)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, D, rhs, α, β)
    mul!(resDo, Do, rhs, α, β)
    mul!(resDo2, Do2, rhs, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))
    resD = rand(T, 100)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, transpose(D), rhs2, α, β)
    mul!(resDo, transpose(Do), rhs2, α, β)
    mul!(resDo2, transpose(Do2), rhs2, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))
    resD = rand(T, 100)
    resDo, resDo2 = copy(resD), copy(resD)
    mul!(resD, adjoint(D), rhs2, α, β)
    mul!(resDo, adjoint(Do), rhs2, α, β)
    mul!(resDo2, adjoint(Do2), rhs2, α, β)
    @test(norm(resD - resDo) <= rtol * norm(resD))
    @test(norm(resD - resDo2) <= rtol * norm(resD))

    K = [opEye(2) opZeros(2, 3); opZeros(3, 2) opEye(3)]
    v = simple_vector(Float64, 5)
    @test all(v .== K * v)

    K = [opEye(2); sparse(1.0I, 2, 2)]
    v = simple_vector(Float64, 2)
    @test all(K * v .== [v; v])
  end
end

test_cat()
