@testset "estimate_opnorm and TSVD tests (LinearOperators)" begin
  # 1) Square Float64 via direct LAPACK or ARPACK
  A_mat = [2.0 0.0; 0.0 -1.0]
  A_op = LinearOperator(A_mat)
  λ, ok = estimate_opnorm(A_op)
  @test ok == true
  @test isapprox(λ, 2.0; rtol = 1e-12)

  # 2) Rectangular Float64 via direct LAPACK or ARPACK SVD
  J_mat = [3.0 0.0 0.0; 0.0 1.0 0.0]
  J_op = LinearOperator(J_mat)
  σ, ok_sv = estimate_opnorm(J_op)
  @test ok_sv == true
  @test isapprox(σ, 3.0; rtol = 1e-12)

  # 3) Square BigFloat via TSVD
  B_mat = Matrix{BigFloat}([2.0 0.0; 0.0 -1.0])
  B_op = LinearOperator(B_mat)
  λ_bf, ok_bf = estimate_opnorm(B_op)
  @test ok_bf == true
  @test isapprox(λ_bf, BigFloat(2); rtol = 1e-12)

  # 4) Rectangular BigFloat via rectangular TSVD
  JR_mat = Matrix{BigFloat}([3.0 0.0 0.0; 0.0 1.0 0.0])
  JR_op = LinearOperator(JR_mat)
  σ_bf, ok_bf2 = estimate_opnorm(JR_op)
  @test ok_bf2 == true
  @test isapprox(σ_bf, BigFloat(3); rtol = 1e-12)
end
