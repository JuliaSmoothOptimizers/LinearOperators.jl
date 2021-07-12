function test_lbfgs()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  # test limited-memory BFGS
  @testset ExtendedTestSet "LBFGS" begin
    n = 10
    mem = 5
    B = LBFGSOperator(n, mem = mem, scaling = false)
    H = InverseLBFGSOperator(n, mem = mem, scaling = false)

    for t = 1:2 # Run again after reset!
      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test B.data.insert == 1
      @test H.data.insert == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ
      @test norm(Matrix(H) - Matrix(1.0I, n, n)) <= ϵ

      # Test that nonpositive curvature can't be added.
      s = simple_vector(Float64, n)
      z = zeros(n)
      push!(B, s, -s)
      @test B.data.insert == 1
      push!(B, s, z)
      @test B.data.insert == 1
      push!(H, s, -s)
      @test H.data.insert == 1
      push!(H, s, z)
      @test H.data.insert == 1

      # Insert a few {s,y} pairs.
      insert = 0
      for i = 1:(mem + 2)
        s = ones(n) * i
        y = [i; ones(n - 1)]
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
      v = simple_vector(Float64, n)
      @test norm(B * v - v) > rtol
      @test norm(H * v - v) > rtol
      reset!(B)
      reset!(H)
      @test B.data.scaling_factor == 1.0
      @test H.data.scaling_factor == 1.0
      @test norm(B * v - v) < rtol
      @test norm(H * v - v) < rtol
    end

    # test against full BFGS without scaling
    mem = n
    LB = LBFGSOperator(n, mem = mem, scaling = false)
    B = Matrix(1.0I, n, n)

    function bfgs!(B, s, y, damped = false)
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

    for k = 1:mem
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      B = bfgs!(B, s, y)
      LB = push!(LB, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end

    # test damped L-BFGS
    B = LBFGSOperator(n, mem = mem, damped = true, scaling = false, σ₂ = 0.8, σ₃ = Inf)
    H = InverseLBFGSOperator(n, mem = mem, damped = true, scaling = false, σ₂ = 0.8, σ₃ = Inf)

    insert_B = insert_H = 0
    for i = 1:(mem + 2)
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      ys = dot(y, s)
      g = simple_vector(Float64, n)
      d = -(H * g) # added brackets here
      α = i / mem
      s = α * d
      if ys > 0.2 * dot(s, B * s)
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
    LB = LBFGSOperator(n, mem = mem, damped = true, scaling = false)
    B = Matrix(1.0I, n, n)

    @test norm(Matrix(LB) - B) < rtol * norm(B)
    @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

    for k = 1:mem
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      B = bfgs!(B, s, y, true)
      LB = push!(LB, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    end
  end

  @testset ExtendedTestSet "Different precision" begin
    n = 10
    mem = 5
    for T in (Float16, Float32, Float64, BigFloat)
      B = LBFGSOperator(T, n, mem = mem)
      H = InverseLBFGSOperator(T, n, mem = mem)
      s = ones(T, n)
      y = ones(T, n)
      push!(B, s, y)
      push!(H, s, y)
      @test eltype(B) == T
      @test eltype(H) == T
      v = [-(-one(T))^i for i = 1:n]
      @test eltype(B * v) == T
      @test eltype(H * v) == T
    end
  end

  @testset "LBFGS allocations" begin
    n = 100
    mem = 20
    B = LBFGSOperator(n, mem = mem)
    H = InverseLBFGSOperator(n, mem = mem)
    BD = LBFGSOperator(n, mem = mem, damped = true)
    HD = InverseLBFGSOperator(n, mem = mem, damped = true)
    for _ = 1:2:n
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
      push!(H, s, y)
      g = rand(n)
      push!(BD, s, y)
      push!(HD, s, y, 1.0, g)
    end
    x = rand(n)
    y = rand(n)
    res = similar(x)
    mul!(res, B, x)  # warmup
    nallocs = @allocated mul!(res, B, x)
    @test nallocs == 0
    mul!(res, H, x)  # warmup
    nallocs = @allocated mul!(res, H, x)
    @test nallocs == 0
    nallocs = @allocated diag!(B, x)
    @test nallocs == 0

    nallocs = @allocated push!(B, x, y)
    @test nallocs == 0
    nallocs = @allocated push!(H, x, y)
    @test nallocs == 0

    tmp = zeros(n)
    nallocs = @allocated push!(BD, x, y, tmp)
    @test nallocs == 0
    nallocs = @allocated push!(HD, x, y, 1.0, x, tmp)
    @test nallocs == 0
  end

  @testset "push! errors" begin
    n = 100
    mem = 20
    B = LBFGSOperator(n, mem = mem)
    H = InverseLBFGSOperator(n, mem = mem)
    BD = LBFGSOperator(n, mem = mem, damped = true)
    HD = InverseLBFGSOperator(n, mem = mem, damped = true)
    s = ones(n)
    y = ones(n)
    g = ones(n)
    Bs = zeros(n)
    @test_throws ErrorException push!(B, s, y, Bs)
    @test_throws ErrorException push!(H, s, y, Bs)
    @test_throws ErrorException push!(HD, s, y, Bs)
    @test_throws ErrorException push!(B, s, y, 1.0, g)
    @test_throws ErrorException push!(BD, s, y, 1.0, g)
    @test_throws ErrorException push!(H, s, y, 1.0, g)
    @test_throws ErrorException push!(B, s, y, 1.0, g, Bs)
    @test_throws ErrorException push!(BD, s, y, 1.0, g, Bs)
    @test_throws ErrorException push!(H, s, y, 1.0, g, Bs)
  end

  @testset "LBFGS eigenvalues" begin
    n = 100
    mem = 20
    B = LBFGSOperator(n, mem = mem)
    H = InverseLBFGSOperator(n, mem = mem)
    for _ = 1:2:n
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
      push!(H, s, y)
    end
    λ = eigs(B, nev = n)[1]  # call Arpack
    @test minimum(λ) > 0
    λ = eigs(H)[1]
    @test minimum(λ) > 0
  end
end

test_lbfgs()