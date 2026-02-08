@testset "CompressedLBFGSOperator operator" begin
  iter=50
  n=100
  n=5
  types = [Float32, Float64]
  for T in types  
    lbfgs = CompressedLBFGSOperator(n; T) # mem=5
    V = LinearOperators.default_vector_type(;T)
    Bv = V(rand(T, n))
    s = V(rand(T, n))
    mul!(Bv, lbfgs, s) # warm-up
    for i in 1:iter
      s = V(rand(T, n))
      y = V(rand(T, n))
      push!(lbfgs, s, y)
      allocs = @allocated mul!(Bv, lbfgs, s)
      @test allocs == 0
      @test Bv ≈ y
    end  
  end
end
