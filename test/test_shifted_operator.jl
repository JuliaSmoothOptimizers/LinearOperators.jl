@testset "ShiftedOperator Tests" begin
  @testset "Real Symmetric (Float64)" begin
    n = 5
    H_dense = rand(n, n)
    H_dense = H_dense + H_dense' # Make it symmetric

    # IMPORTANT: You must explicitly tell LinearOperator it is symmetric
    H_op = LinearOperator(H_dense; symmetric = true, hermitian = true)

    σ = 2.0
    op = ShiftedOperator(H_op, σ)

    # Reference: A_ref = H + σI
    A_ref = H_dense + σ * I

    x = rand(n)
    y = zeros(n)

    # 1. Test Properties
    @test size(op) == (n, n)
    @test issymmetric(op) == true
    @test ishermitian(op) == true
    # Check internal data access
    @test op.data.σ == σ

    # 2. Test 3-arg mul! (y = op * x)
    mul!(y, op, x)
    @test y ≈ A_ref * x

    # 3. Test 5-arg mul! (y = α * op * x + β * y)
    y_orig = rand(n)
    y = copy(y_orig)
    α, β = 0.5, -1.0
    mul!(y, op, x, α, β)
    @test y ≈ α * (A_ref * x) + β * y_orig

    # 4. Test Transpose
    # Should return a new ShiftedOperator
    op_t = transpose(op)
    @test op_t isa ShiftedOperator

    y_t = zeros(n)
    mul!(y_t, op_t, x)
    @test y_t ≈ transpose(A_ref) * x
  end

  @testset "Complex Non-Hermitian" begin
    n = 5
    H_dense = rand(ComplexF64, n, n)
    # Not symmetric or hermitian
    H_op = LinearOperator(H_dense)

    σ = 1.0 + 2.0im
    op = ShiftedOperator(H_op, σ)

    A_ref = H_dense + σ * I
    x = rand(ComplexF64, n)

    # 1. Test Adjoint
    op_c = adjoint(op)
    @test op_c isa ShiftedOperator
    # Check that the shift inside the adjoint data is conjugated
    @test op_c.data.σ == conj(σ)

    y_c = zeros(ComplexF64, n)
    mul!(y_c, op_c, x)
    @test y_c ≈ adjoint(A_ref) * x
  end

  @testset "Mutation (Updating Sigma)" begin
    n = 3
    H_dense = rand(n, n)
    # Use simple matrix as inner operator
    op = ShiftedOperator(LinearOperator(H_dense), 1.0)

    x = ones(n)
    y1 = op * x

    # Update the shift via the data field
    op.data.σ = 10.0

    y2 = op * x

    # Verify the result changed and matches the new shift
    @test !(y1 ≈ y2)
    @test y2 ≈ (H_dense + 10.0*I) * x
  end
end
