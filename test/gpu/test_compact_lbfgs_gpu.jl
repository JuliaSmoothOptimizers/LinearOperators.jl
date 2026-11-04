using Test, LinearOperators, LinearAlgebra

# Run CompactLBFGSOperator with GPU storage and compare it with the same operator on the CPU.
# Scalar indexing should be disallowed by the caller so that it is detected.
function test_compact_lbfgs_gpu(; arrayType, T = Float32)
  @testset "CompactLBFGSOperator with arrayType $(arrayType) and $(T)" begin
    n = 32
    S = typeof(arrayType(zeros(T, n)))
    rtol = sqrt(eps(T))
    for mem in (1, 3, 5), scaling in (true, false), damped in (false, true)
      kwargs = (mem = mem, scaling = scaling, damped = damped)
      C = CompactLBFGSOperator(T, n; kwargs..., S = S) # GPU 
      R = CompactLBFGSOperator(T, n; kwargs...)        # non-GPU
      @test LinearOperators.storage_type(C) == S
      @test C.data.Sk isa typeof(arrayType(zeros(T, n, mem)))

      for i = 1:(2 * mem + 2)
        s, y = randn(T, n), randn(T, n)
        push!(C, arrayType(s), arrayType(y))
        push!(R, s, y)
        @test C.data._insert == R.data._insert
        x = randn(T, n)
        Cx = C * arrayType(x)
        @test Cx isa S
        @test isapprox(Array(Cx), R * x; rtol = rtol)
      end

      reset!(C)
      x = randn(T, n)
      @test isapprox(Array(C * arrayType(x)), C.data.ξ * x; rtol = rtol)
    end
  end
end
