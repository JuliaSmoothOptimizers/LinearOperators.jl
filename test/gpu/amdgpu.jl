using Test, LinearAlgebra, SparseArrays
using LinearOperators, AMDGPU

@testset "AMDGPU -- AMDGPU.jl" begin
  A = ROCArray(rand(Float32, 5, 5))
  B = ROCArray(rand(Float32, 10, 10))
  C = ROCArray(rand(Float32, 20, 20))
  M = BlockDiagonalOperator(A, B, C)
  v = ROCArray(rand(5))


  v = ROCArray(rand(Float32, 35))
  y = M * v
  @test y isa ROCArray{Float32}

  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(adjoint(A))
  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(transpose(A))
  @test LinearOperators.storage_type(A) == LinearOperators.storage_type(adjoint(A))
  @test LinearOperators.storage_type(Diagonal(v)) == typeof(v)


  @testset "AMDGPU S kwarg" test_S_kwarg(arrayType = ROCArray)
end
