function test_lbfgs()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  # test limited-memory BFGS
  @testset "LBFGS" begin
    n = 10
    mem = 5
    B = LBFGSOperator(n, mem, scaling=false)
    H = InverseLBFGSOperator(n, mem, scaling=false)

    for t = 1:2 # Run again after reset!
      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test B.data.insert == 1
      @test H.data.insert == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ
      @test norm(Matrix(H) - Matrix(1.0I, n, n)) <= ϵ

      # Test that nonpositive curvature can't be added.
      s = rand(n)
      z = zeros(n)
      push!(B, s, -s); @test B.data.insert == 1
      push!(B, s,  z); @test B.data.insert == 1
      push!(H, s, -s); @test H.data.insert == 1
      push!(H, s,  z); @test H.data.insert == 1

      # Insert a few {s,y} pairs.
      insert = 0
      for i = 1 : mem+2
        s = rand(n)
        y = rand(n)
        if dot(s, y) > 1.0e-20
          insert += 1
          push!(B, s, y)
          push!(H, s, y)
        end
      end

      @test B.data.insert == mod(insert, B.data.mem) + 1
      @test H.data.insert == mod(insert, H.data.mem) + 1

      @test check_positive_definite(B)
      @test check_positive_definite(H)

      @test check_hermitian(B)
      @test check_hermitian(H)

      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test norm(Matrix(H * B) - Matrix(1.0I, n, n)) <= rtol

      # Testing reset! function
      v = rand(n)
      @test norm(B * v - v) > rtol
      @test norm(H * v - v) > rtol
      reset!(B)
      reset!(H)
      @test norm(B * v - v) < rtol
      @test norm(H * v - v) < rtol
    end

    # test against full BFGS without scaling
    mem = n
    LB = LBFGSOperator(n, mem, scaling=false)
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
      s = rand(n)
      y = rand(n)
      B = bfgs!(B, s, y)
      LB = push!(LB, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end

    # test damped L-BFGS
    B = LBFGSOperator(n, mem, damped=true, scaling=false)
    H = InverseLBFGSOperator(n, mem, damped=true, scaling=false)

    insert_B = insert_H = 0
    for i = 1 : mem+2
      s = rand(n)
      y = rand(n)
      ys = dot(y, s)
      g = rand(n)
      d = -H * g
      α = rand()
      s = α * d
      if ys > B.data.damp_factor * dot(s, B * s) && ys > B.data.damp_factor * dot(y, H * y)
        insert_B += 1
        insert_H += 1
        push!(B, s, y)
        push!(H, s, y, α, g)
      end
    end

    @test B.data.insert == mod(insert_B, B.data.mem) + 1
    @test H.data.insert == mod(insert_H, H.data.mem) + 1

    @test check_positive_definite(B)
    @test check_positive_definite(H)

    @test check_hermitian(B)
    @test check_hermitian(H)

    @test norm(diag(B) - diag(Matrix(B))) <= rtol

    @test norm(Matrix(H * B) - Matrix(1.0I, n, n)) <= rtol

    # test against full BFGS without scaling
    mem = n
    LB = LBFGSOperator(n, mem, damped=true, scaling=false)
    B = Matrix(1.0I, n, n)

    @test norm(Matrix(LB) - B) < rtol * norm(B)
    @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

    for k = 1 : mem
      s = rand(n)
      y = rand(n)
      B = bfgs!(B, s, y, true)
      LB = push!(LB, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end
  end
end

test_lbfgs()
