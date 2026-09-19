function test_lqn()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  @testset ExtendedTestSet "LQN" begin
    n = 10
    mem = 5
    B = LQNOperator(n, mem = mem, scaling = false)
    @test isallocated5(B) == true

    for t = 1:2
      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test B.data.insert == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ

      # Insert a few {s,y} pairs, alternating the sign of y to exercise both the BFGS
      # and the SR1 branch of the automatic update selection.
      for i = 1:(mem + 2)
        s = ones(n) * i
        y = isodd(i) ? [i; ones(n - 1)] : -[i; ones(n - 1)]
        push!(B, s, y)
      end

      @test check_hermitian(B)
      @test norm(diag(B) - diag(Matrix(B))) <= rtol
      @test any(t -> t !== :empty, B.data.update_type)

      v = simple_vector(Float64, n)
      @test norm(B * v - v) > rtol
      reset!(B)
      @test B.data.scaling_factor == 1.0
      @test all(t -> t === :empty, B.data.update_type)
      @test norm(B * v - v) < rtol

      # Test upper bound
      @test opnorm(Matrix(B)) ≤ B.data.opnorm_upper_bound
    end

    @testset "Automatic BFGS/SR1 selection matches a hand-derived reference" begin
      # B₀ = I. Each pair below is chosen so that the sign of the curvature sᵀy alternates,
      # forcing the operator to alternate between a BFGS and an SR1 update.
      n = 4
      LB = LQNOperator(n, mem = n, scaling = false)

      push!(LB, [1.0, 0, 0, 0], [2.0, 0, 0, 0])   # ys = 2 > 0  -> BFGS
      push!(LB, [0.0, 1, 0, 0], [0.0, -1, 0, 0])  # ys = -1 < 0 -> SR1
      push!(LB, [0.0, 0, 1, 0], [0.0, 0, 3, 0])   # ys = 3 > 0  -> BFGS
      push!(LB, [0.0, 0, 0, 1], [0.0, 0, 0, -2.0]) # ys = -2 < 0 -> SR1

      @test LB.data.update_type == [:bfgs, :sr1, :bfgs, :sr1]
      Bref = Diagonal([2.0, -1.0, 3.0, -2.0])
      @test norm(Matrix(LB) - Bref) < rtol
      @test norm(diag(LB) - diag(Bref)) < rtol
      @test check_hermitian(LB)
    end

    @testset "Reject pairs for which neither update is well defined" begin
      n = 5
      B = LQNOperator(n, mem = 3, scaling = false)
      s = simple_vector(Float64, n)
      # y = B*s = s makes the SR1 residual r = y - B*s = 0, which is not well defined,
      # while the curvature sᵀy = ‖s‖² > 0 does allow a (trivial) BFGS update.
      push!(B, s, s)
      @test B.data.insert == 2
      @test B.data.update_type[1] == :bfgs

      # y = -s gives negative curvature, and the SR1 residual r = y - B*s = -2s is
      # collinear with s, so sᵀr = -2‖s‖² ≠ 0: SR1 should be accepted.
      push!(B, s, -s)
      @test B.data.insert == 3
      @test B.data.update_type[2] == :sr1

      # A zero step should be rejected by both updates.
      z = zeros(n)
      push!(B, z, z)
      @test B.data.insert == 3
    end

    # test against a dense reference that mimics the same BFGS/SR1 selection rule,
    # without scaling and using the full memory (so that no pair is ever evicted).
    n = 6
    mem = n
    LB = LQNOperator(n, mem = mem, scaling = false)
    Bd = Matrix(1.0I, n, n)

    function lqn!(Bd, s, y)
      Bs = Bd * s
      sBs = dot(s, Bs)
      ys = dot(y, s)
      sNorm = norm(s)
      if ys ≥ ϵ + ϵ * norm(y) * sNorm && sBs ≥ ϵ + ϵ * norm(Bs) * sNorm
        Bd = Bd + y * y' / ys - Bs * Bs' / sBs
      else
        r = y - Bs
        as = dot(s, r)
        if abs(as) ≥ ϵ + ϵ * norm(r) * sNorm
          Bd = Bd + r * r' / as
        end
      end
      return Bd
    end

    for k = 1:mem
      s = rand(n) .- 0.5
      y = isodd(k) ? rand(n) : -rand(n)
      Bd = lqn!(Bd, s, y)
      push!(LB, s, y)
      @test norm(Matrix(LB) - Bd) < rtol * max(1, norm(Bd))
      @test norm(diag(LB) - diag(Bd)) < rtol * max(1, norm(diag(Bd)))
    end

    # Test upper bound
    @test opnorm(Bd) ≤ LB.data.opnorm_upper_bound
  end

  @testset ExtendedTestSet "Different precision" begin
    n = 10
    mem = 5
    for T in (Float16, Float32, Float64, BigFloat)
      B = LQNOperator(T, n, mem = mem)
      s = ones(T, n)
      y = ones(T, n)
      push!(B, s, y)
      @test eltype(B) == T
      v = [-(-one(T))^i for i = 1:n]
      @test eltype(B * v) == T
    end
  end

  @testset "LQN allocations" begin
    n = 100
    mem = 20
    B = LQNOperator(n, mem = mem)
    nallocs = 0
    for _ = 1:2:n
      s = rand(n)
      y = isodd(rand(1:2)) ? rand(n) : -rand(n)
      nallocs += @allocated push!(B, s, y)
    end
    @test nallocs == 0
    x = rand(n)
    res = similar(x)
    mul!(res, B, x)  # warmup
    nallocs = @allocated mul!(res, B, x)
    @test nallocs == 0
    nallocs = @allocated diag!(B, x)
    @test nallocs == 0
  end

  @testset "LQN eigenvalues" begin
    n = 50
    mem = 15
    B = LQNOperator(n, mem = mem)
    for _ = 1:2:n
      s = rand(n)
      y = isodd(rand(1:2)) ? rand(n) : -rand(n)
      push!(B, s, y)
    end
    vals = eigs(B, nev = n - 1)
    resid = vals[end]
    @test norm(resid) ≤ sqrt(eps(eltype(B))) * n
  end
end

test_lqn()
