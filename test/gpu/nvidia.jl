using Test, LinearAlgebra, SparseArrays
using LinearOperators, CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

@testset "Nvidia -- CUDA.jl" begin

  @test CUDA.functional()
  CUDA.allowscalar(false)

  A = CUDA.rand(5,5)
  B = CUDA.rand(10,10)
  C = CUDA.rand(20,20)
  M = BlockDiagonalOperator(A, B, C)

  v = CUDA.rand(35)
  y = M * v
  @test y <: CuVector{Float32}
end
