@testset "ShiftedOperator Tests" begin
  @testset "Real Symmetric (Float64)" begin
    n = 5
    H_dense = rand(n, n)
    H_dense = H_dense + H_dense'

    H_op = LinearOperator(H_dense; symmetric = true, hermitian = true)

    σ = 2.0
    op = ShiftedOperator(H_op, σ)

    A_ref = H_dense + σ * I

    x = rand(n)
    y = zeros(n)

    @test size(op) == (n, n)
    @test issymmetric(op) == true
    @test ishermitian(op) == true
    @test op.data.σ == σ

    mul!(y, op, x)
    @test y ≈ A_ref * x

    y_orig = rand(n)
    y = copy(y_orig)
    α, β = 0.5, -1.0
    mul!(y, op, x, α, β)
    @test y ≈ α * (A_ref * x) + β * y_orig

    op_t = transpose(op)

    y_t = zeros(n)
    mul!(y_t, op_t, x)
    @test y_t ≈ transpose(A_ref) * x
  end

  @testset "Complex Non-Hermitian" begin
    n = 5
    H_dense = rand(ComplexF64, n, n)
    H_op = LinearOperator(H_dense)

    σ = 1.0 + 2.0im
    op = ShiftedOperator(H_op, σ)

    A_ref = H_dense + σ * I
    x = rand(ComplexF64, n)

    op_c = adjoint(op)

    y_c = zeros(ComplexF64, n)
    mul!(y_c, op_c, x)
    @test y_c ≈ adjoint(A_ref) * x
  end

  @testset "Mutation (Updating Sigma)" begin
    n = 3
    H_dense = rand(n, n)
    op = ShiftedOperator(LinearOperator(H_dense), 1.0)

    x = ones(n)
    y1 = op * x

    op.data.σ = 10.0

    y2 = op * x

    @test !(y1 ≈ y2)
    @test y2 ≈ (H_dense + 10.0*I) * x
  end

  @testset "Strict Type Constraint" begin
    H = LinearOperator(rand(Float32, 5, 5))
    σ = 1.0
    op = ShiftedOperator(H, σ)

    @test eltype(op) == Float32
    @test op.data.σ isa Float32

    x = rand(Float32, 5)
    y = op * x
    @test eltype(y) == Float32
  end
end
