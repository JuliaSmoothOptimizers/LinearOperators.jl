function test_compactlbfgs()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  # test limited-memory BFGS
  @testset "Compact LBFGS" begin
    n = 10
    mem = 5
    B = CompactLBFGSOperator(n, mem)

    for t = 1:2 # Run again after reset!
      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test B.data.col == 0
      @test B.data.head == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ

      # Test that nonpositive curvature can't be added.
      s = simple_vector(Float64, n)
      z = zeros(n)
      push!(B, s, -s); @test B.data.col == 0
      push!(B, s,  z); @test B.data.col == 0

      # Insert a few {s,y} pairs.
      col = 0
      for i = 1 : mem+2
        s = ones(n) * i
        y = [i; ones(n-1)]
        if dot(s, y) > 1.0e-20
          col += 1
          push!(B, s, y)
        end
      end

      @test B.data.head == mod(col, B.data.mem) + 1

      @test check_positive_definite(B)

      @test check_hermitian(B)

      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      # Testing reset! function
      v = simple_vector(Float64, n)
      @test norm(B * v - v) > rtol
      reset!(B)
      @test norm(B * v - v) < rtol
    end

    # test against dense BFGS without scaling
    mem = n
    LB = CompactLBFGSOperator(n, mem)

    slast = simple_vector(Float64, n)
    ylast = simple_vector(Float64, n)

    B = Matrix(1.0I, n, n)

    function bfgs!(B, s, y, damped=false)
      # dense BFGS update
      ys = dot(y, s)
      Bs = B * s
      tol = damped ? (0.2 * dot(s, Bs)) : 1.0e-20
      if ys > tol
        B = B - Bs * Bs' / dot(s, Bs) + y * y' / ys
      end
      return B
    end

    @test norm(Matrix(LB) - B) < rtol * norm(B)
    @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

    for k = 1 : mem
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      B = bfgs!(B, s, y)
      LB = push!(LB, s, y)
      @test LB.data.θ == dot(y, y) / dot(y, s)
      LB.data.θ = 1.0 # Remove scaling for the test against dense BFGS
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end
  end

  @testset "Different precision" begin
    n = 10
    mem = 5
    for T in (Float16, Float32, Float64, BigFloat)
      B = CompactLBFGSOperator(T, n, mem)
      s = ones(T, n)
      y = ones(T, n)
      push!(B, s, y)
      @test eltype(B) == T
      v = [-(-one(T))^i for i = 1:n]
      @test eltype(B * v) == T
    end
  end
end

test_compactlbfgs()
