using Test, LinearAlgebra, SparseArrays
using LinearOperators, CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

@testset "Nvidia -- CUDA.jl" begin
  @test CUDA.functional()
  CUDA.allowscalar(false)

  A = CUDA.rand(5, 5)
  B = CUDA.rand(10, 10)
  C = CUDA.rand(20, 20)
  M = BlockDiagonalOperator(A, B, C)

  v = CUDA.rand(35)
  y = M * v
  @test y isa CuVector{Float32}

  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(adjoint(A))
  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(transpose(A))
  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(adjoint(A))
  @test LinearOperators.storage_type(Diagonal(v)) == typeof(v)

  @testset "Nvidia S kwarg" test_S_kwarg(arrayType = CuArray)
end
