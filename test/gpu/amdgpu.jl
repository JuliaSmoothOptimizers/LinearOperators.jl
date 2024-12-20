using Test, LinearAlgebra, SparseArrays
using LinearOperators, AMDGPU

@testset "AMDGPU -- AMDGPU.jl" begin
  A = ROCArray(rand(Float32, 5, 5))
  B = ROCArray(rand(Float32, 10, 10))
  C = ROCArray(rand(Float32, 20, 20))
  M = BlockDiagonalOperator(A, B, C)

  v = ROCArray(rand(Float32, 35))
  y = M * v
  @test y isa ROCArray{Float32}

  @testset "AMDGPU S kwarg" test_S_kwarg(arrayType = ROCArray)
end
