function test_lsr1()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  @testset "LSR1" begin
    n = 10
    mem = 5
    B = LSR1Operator(n, mem, scaling=false)

    for t = 1:2
      @assert norm(diag(B) - diag(Matrix(B))) <= rtol

      @assert B.data.insert == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ

      # Test that only valid updates are accepted.
      s = rand(n)
      y = B * s
      push!(B, s, y); @assert B.data.insert == 1

      # Insert a few {s,y} pairs.
      for i = 1 : mem+2
        s = rand(n)
        y = rand(n)
        push!(B, s, y)
      end

      @test check_hermitian(B)
      @assert norm(diag(B) - diag(Matrix(B))) <= rtol

      v = rand(n)
      @assert norm(B * v - v) > rtol
      reset!(B)
      @assert norm(B * v - v) < rtol
    end

    # test against full SR1 without scaling
    mem = n
    LB = LSR1Operator(n, mem, scaling=false)
    B = Matrix(1.0I, n, n)

    function sr1!(B, s, y)
      # dense SR1 update
      ymBs = y - B * s
      denom = dot(ymBs, s)
      if abs(denom) >= 1.0e-8 + 1.0e-8 * norm(s) * norm(ymBs)
        B = B + ymBs * ymBs' / denom
      end
      return B
    end

    @assert norm(Matrix(LB) - B) < rtol * norm(B)
    @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

    for k = 1 : mem
      s = rand(n)
      y = rand(n)
      B = sr1!(B, s, y)
      LB = push!(LB, s, y)
      @assert norm(Matrix(LB) - B) < rtol * norm(B)
      @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end
  end

  @testset "Different precision" begin
    n = 10
    mem = 5
    for T in (Float16, Float32, Float64, BigFloat)
      B = LSR1Operator(T, n, mem)
      s = ones(T, n)
      y = ones(T, n)
      push!(B, s, y)
      @test eltype(B) == T
      v = [-(-one(T))^i for i = 1:n]
      @test eltype(B * v) == T
    end
  end
end

test_lsr1()
