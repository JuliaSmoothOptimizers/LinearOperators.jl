using TSVD, GenericLinearAlgebra
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

    @test !issymmetric(op)
    @test !ishermitian(op)

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

  @testset "Mutation (Dynamic Hermitian Check)" begin
    n = 3
    H_dense = rand(ComplexF64, n, n)
    H_dense = H_dense + H_dense'
    H_op = LinearOperator(H_dense; symmetric = false, hermitian = true)

    σ = 2.0
    op = ShiftedOperator(H_op, σ)
    @test ishermitian(op)

    op.data.σ = 2.0 + 1.0im
    @test !ishermitian(op)

    op.data.σ = 3.0
    @test ishermitian(op)
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

  @testset "Coverage & Utilities" begin
    n = 5
    H = LinearOperator(rand(n, n))
    x = rand(n)
    y = zeros(n)
    op = ShiftedOperator(H, 2.0)

    mul!(y, transpose(op), x, 0.5, 1.0)

    mul!(y, transpose(op), x, 0.0, 1.0)

    op_zero = ShiftedOperator(H, 0.0)
    mul!(y, transpose(op_zero), x)

    @test LinearOperators.isallocated5(op) == true

    @test LinearOperators.storage_type(op) == LinearOperators.storage_type(H)

    op.nprod = 10
    reset!(op)
    @test op.nprod == 0
  end

  @testset "OpNorm Fallback & Error Handling Coverage" begin
    """
            make_mock_throwing_op(m, n, is_sym, is_herm, err_msg)

        Creates a "mock" LinearOperator designed specifically to fail when applied to a vector.
        
        Why is this needed?
        To get CodeCov on our `catch` blocks and Krylov dimension (NCV) retry loops, we need 
        ARPACK to fail with specific Fortran errors (like "AUPD"). Triggering these errors 
        naturally using real math requires highly pathological matrices, which causes "flaky" 
        tests that might pass on one machine and fail on another (e.g., in CI pipelines).

        By using a mock operator that instantly throws an error containing our target string, 
        we guarantee 100% stable test coverage of our exception handling and retry logic.
        """
    function make_mock_throwing_op(m, n, is_sym, is_herm, err_msg)
      return LinearOperator(
        Float64,
        m,
        n,
        is_sym,
        is_herm,
        (args...) -> throw(ErrorException(err_msg)),
        (args...) -> throw(ErrorException(err_msg)),
        (args...) -> throw(ErrorException(err_msg)),
      )
    end

    @testset "Dense Branch Success" begin
      # Size <= 5 so it tries dense and succeeds. (Covers the successful try block)
      op_sym_small = LinearOperator(Symmetric(rand(4, 4)))
      val_sym, succ_sym = estimate_opnorm(op_sym_small; tiny_dense_threshold = 5)
      @test succ_sym

      op_nonsym_small = LinearOperator(rand(4, 4))
      val_nonsym, succ_nonsym = estimate_opnorm(op_nonsym_small; tiny_dense_threshold = 5)
      @test succ_nonsym
    end

    @testset "Dense Fallback & Generic Error Rethrow" begin
      # Size <= 5 so it tries dense. Dense calls Matrix(), which applies the operator and throws.
      # Caught by dense catch -> falls back to iterative -> applies operator and throws again.
      # Since the error isn't an ARPACK/AUPD error, it hits `else rethrow(e)`.
      op_err_sym = make_mock_throwing_op(4, 4, true, true, "Generic Error")
      @test_throws ErrorException estimate_opnorm(op_err_sym; tiny_dense_threshold = 5)

      op_err_nonsym = make_mock_throwing_op(4, 4, false, false, "Generic Error")
      @test_throws ErrorException estimate_opnorm(op_err_nonsym; tiny_dense_threshold = 5)
    end

    @testset "opnorm_eig: ARPACK/AUPD Logic" begin
      # Simulate AUPD failure: NCV increases, max_attempts exhausts -> return NaN, false
      # Starts ncv=20. We use n=100.
      op_aupd_sym_large = make_mock_throwing_op(100, 100, true, true, "Mock AUPD Error")
      @test_logs (:warn, r"opnorm_eig: increasing NCV from 20 to 40") match_mode=:any begin
        val, succ = estimate_opnorm(op_aupd_sym_large; tiny_dense_threshold = 5, max_attempts = 2)
        @test isnan(val)
        @test !succ
      end

      # Simulate AUPD failure: NCV hits maximum limit (n) -> logs and rethrows
      # Starts ncv=20. We use n=25.
      # Attempt 1: throws, ncv becomes 25. 
      # Attempt 2: throws, ncv >= n (25 >= 25) -> logs warning and rethrows!
      op_aupd_sym_small = make_mock_throwing_op(25, 25, true, true, "Mock AUPD Error")
      @test_logs (:warn, r"opnorm_eig: increasing NCV from 20 to 25") (
        :warn,
        r"ARPACK failed and NCV cannot be increased further.",
      ) match_mode=:any begin
        @test_throws ErrorException estimate_opnorm(
          op_aupd_sym_small;
          tiny_dense_threshold = 5,
          max_attempts = 3,
        )
      end
    end

    @testset "TSVD Fallback for BigFloat" begin
      op_bigfloat = LinearOperator(rand(BigFloat, 4, 4))
      val, succ = estimate_opnorm(op_bigfloat; tiny_dense_threshold = 5)
      @test succ  # TSVD-based fallback should succeed.
    end

    @testset "opnorm_svd: ARPACK/AUPD Logic" begin
      # Same logic for opnorm_svd (non-Hermitian).
      # Starts ncv=10. We use n=50. Max attempts=2.
      op_aupd_nonsym_large = make_mock_throwing_op(50, 50, false, false, "Mock AUPD Error")
      @test_logs (:warn, r"opnorm_svd: increasing NCV from 10 to 20") match_mode=:any begin
        val, succ =
          estimate_opnorm(op_aupd_nonsym_large; tiny_dense_threshold = 5, max_attempts = 2)
        @test isnan(val)
        @test !succ
      end

      # Starts ncv=10. n=15. Hits NCV limit and rethrows.
      op_aupd_nonsym_small = make_mock_throwing_op(15, 15, false, false, "Mock AUPD Error")
      @test_logs (:warn, r"opnorm_svd: increasing NCV from 10 to 15") (
        :warn,
        r"ARPACK failed and NCV cannot be increased further.",
      ) match_mode=:any begin
        @test_throws ErrorException estimate_opnorm(
          op_aupd_nonsym_small;
          tiny_dense_threshold = 5,
          max_attempts = 3,
        )
      end
    end
  end
end
