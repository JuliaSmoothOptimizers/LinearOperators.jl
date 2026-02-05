using TSVD
using Arpack
using GenericLinearAlgebra
@testset "estimate_opnorm type stability and dispatch" begin
  for T in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "Type $T" begin
      
      # A) Rectangular Matrix (Forces SVD path)
      J_mat = zeros(T, 3, 2)
      J_mat[1, 1] = 3.0
      J_mat[2, 2] = 1.0
      J_op = LinearOperator(J_mat)
      
      σ, ok = estimate_opnorm(J_op)
      @test ok
      @test isapprox(σ, 3.0; rtol=1e-5)

      # B) Hermitian Wrapper (Forces Eig path via dispatch)
      H_data = rand(T, 10, 10)
      H_data = H_data + H_data' 
      H = Hermitian(H_data)
      
      exact_norm = opnorm(H) 
      est_norm, ok = estimate_opnorm(H)
      @test ok
      @test isapprox(est_norm, exact_norm; rtol=1e-5)

      # C) Symmetric Wrapper (Dispatch varies by Real vs Complex)
      S_data = rand(T, 10, 10)
      S_data = S_data + transpose(S_data)
      S = Symmetric(S_data)

      exact_norm_S = opnorm(Matrix(S))
      est_norm_S, ok_S = estimate_opnorm(S)
      @test ok_S
      @test isapprox(est_norm_S, exact_norm_S; rtol=1e-5)
      
      # C.1) Specific check: Ensure Symmetric{Complex} works 
      # (This is NOT Hermitian, so it must successfully fall back to the SVD path)
      if T <: Complex
        @test !ishermitian(S) 
      end
    end
  end
  @testset "BigFloat (Generic)" begin
    B_mat = Matrix{BigFloat}([2.0 0.0; 0.0 -1.0])
    B_op = LinearOperator(B_mat)
    λ_bf, ok_bf = estimate_opnorm(B_op)
    @test ok_bf
    @test isapprox(λ_bf, BigFloat(2); rtol=1e-12)
  end
end