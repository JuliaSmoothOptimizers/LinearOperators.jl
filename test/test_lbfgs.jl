function test_lbfgs()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  # test limited-memory BFGS
  @testset ExtendedTestSet "LBFGS" begin
    n = 10
    mem = 5
    B = LBFGSOperator(n, mem = mem, scaling = false)
    H = InverseLBFGSOperator(n, mem = mem, scaling = false)
    C = CompactLBFGSOperator(n, mem = mem, scaling = false)
    @test isallocated5(B) == isallocated5(H) == isallocated5(C) == true

    for t = 1:2 # Run again after reset!
      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test B.data.insert == 1
      @test H.data.insert == 1
      @test C.data._insert == 1
      @test norm(Matrix(B) - Matrix(1.0I, n, n)) <= ϵ
      @test norm(Matrix(H) - Matrix(1.0I, n, n)) <= ϵ
      @test norm(Matrix(C) - Matrix(1.0I, n, n)) <= ϵ

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
      push!(C, s, -s)
      @test C.data._insert == 1
      push!(C, s, z)
      @test C.data._insert == 1

      # Insert a few {s,y} pairs.
      insert = 0
      for i = 1:(mem + 2)
        s = ones(n) * i
        y = [i; ones(n - 1)]
        if dot(s, y) > 1.0e-20
          insert += 1
          push!(B, s, y)
          push!(H, s, y)
          push!(C, s, y)
        end
      end

      @test B.data.insert == mod(insert, B.data.mem) + 1
      @test H.data.insert == mod(insert, H.data.mem) + 1
      @test C.data._insert == min(insert, C.data._mem) + 1

      @test check_positive_definite(B)
      @test check_positive_definite(H)
      @test check_positive_definite(C)

      @test check_hermitian(B)
      @test check_hermitian(H)
      @test check_hermitian(C)

      @test norm(diag(B) - diag(Matrix(B))) <= rtol

      @test norm(Matrix(H * B) - Matrix(1.0I, n, n)) <= rtol
      @test norm(Matrix(H * C) - Matrix(1.0I, n, n)) <= rtol
      @test norm(Matrix(C) - Matrix(B)) <= rtol * norm(Matrix(B))

      # Testing reset! function
      v = simple_vector(Float64, n)
      @test norm(B * v - v) > rtol
      @test norm(H * v - v) > rtol
      @test norm(C * v - v) > rtol
      reset!(B)
      reset!(H)
      reset!(C)
      @test B.data.scaling_factor == 1.0
      @test H.data.scaling_factor == 1.0
      @test C.data.ξ == 1.0
      @test norm(B * v - v) < rtol
      @test norm(H * v - v) < rtol
      @test norm(C * v - v) < rtol

      # test upper bound
      @test opnorm(Matrix(B)) ≤ B.data.opnorm_upper_bound
    end

    # test against full BFGS without scaling
    mem = n
    LB = LBFGSOperator(n, mem = mem, scaling = false)
    LC = CompactLBFGSOperator(n, mem = mem, scaling = false)
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
    @test norm(Matrix(LC) - B) < rtol * norm(B)

    for k = 1:mem
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      B = bfgs!(B, s, y)
      LB = push!(LB, s, y)
      LC = push!(LC, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
      @test norm(Matrix(LC) - B) < rtol * norm(B)
    end

    # Test upper bound
    @test opnorm(B) ≤ LB.data.opnorm_upper_bound

    # test damped L-BFGS
    B = LBFGSOperator(n, mem = mem, damped = true, scaling = false, σ₂ = 0.8, σ₃ = Inf)
    H = InverseLBFGSOperator(n, mem = mem, damped = true, scaling = false, σ₂ = 0.8, σ₃ = Inf)
    C = CompactLBFGSOperator(n, mem = mem, damped = true, scaling = false, σ₂ = 0.8, σ₃ = Inf)

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
        push!(C, s, y)
      end
    end

    @test B.data.insert == mod(insert_B, B.data.mem) + 1
    @test H.data.insert == mod(insert_H, H.data.mem) + 1
    @test C.data._insert == min(insert_B, C.data._mem) + 1

    @test check_positive_definite(B)
    @test check_positive_definite(H)
    @test check_positive_definite(C)

    @test check_hermitian(B)
    @test check_hermitian(H)
    @test check_hermitian(C)

    @test norm(diag(B) - diag(Matrix(B))) <= rtol

    @test norm(Matrix(H * B) - Matrix(1.0I, n, n)) <= rtol
    @test norm(Matrix(C) - Matrix(B)) <= rtol * norm(Matrix(B))

    # Test upper bound
    @test opnorm(Matrix(B)) ≤ B.data.opnorm_upper_bound

    # test against full BFGS without scaling
    mem = n
    LB = LBFGSOperator(n, mem = mem, damped = true, scaling = false)
    LC = CompactLBFGSOperator(n, mem = mem, damped = true, scaling = false)
    B = Matrix(1.0I, n, n)

    @test norm(Matrix(LB) - B) < rtol * norm(B)
    @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
    @test norm(Matrix(LC) - B) < rtol * norm(B)

    for k = 1:mem
      s = simple_vector(Float64, n)
      y = simple_vector(Float64, n)
      B = bfgs!(B, s, y, true)
      LB = push!(LB, s, y)
      LC = push!(LC, s, y)
      @test norm(Matrix(LB) - B) < rtol * norm(B)
      @test norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
      @test norm(Matrix(LC) - B) < rtol * norm(B)
    end

    # Test upper bound
    @test opnorm(Matrix(B)) ≤ LB.data.opnorm_upper_bound
  end

  @testset ExtendedTestSet "Different precision" begin
    n = 10
    mem = 5
    for T in (Float16, Float32, Float64, BigFloat)
      B = LBFGSOperator(T, n, mem = mem)
      H = InverseLBFGSOperator(T, n, mem = mem)
      C = CompactLBFGSOperator(T, n, mem = mem)
      s = ones(T, n)
      y = ones(T, n)
      push!(B, s, y)
      push!(H, s, y)
      push!(C, s, y)
      @test eltype(B) == T
      @test eltype(H) == T
      @test eltype(C) == T
      v = [-(-one(T))^i for i = 1:n]
      @test eltype(B * v) == T
      @test eltype(H * v) == T
      @test eltype(C * v) == T
    end
  end

  @testset "LBFGS allocations" begin
    n = 100
    mem = 20
    B = LBFGSOperator(n, mem = mem)
    H = InverseLBFGSOperator(n, mem = mem)
    BD = LBFGSOperator(n, mem = mem, damped = true)
    HD = InverseLBFGSOperator(n, mem = mem, damped = true)
    C = CompactLBFGSOperator(n, mem = mem)
    CD = CompactLBFGSOperator(n, mem = mem, damped = true)
    for _ = 1:2:n
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
      push!(H, s, y)
      g = rand(n)
      push!(BD, s, y)
      push!(HD, s, y, 1.0, g)
      push!(C, s, y)
      push!(CD, s, y)
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
    mul!(res, C, x)  # warmup
    nallocs = @allocated mul!(res, C, x)
    @test nallocs == 0
    nallocs = @allocated diag!(B, x)
    @test nallocs == 0

    nallocs = @allocated push!(B, x, y)
    @test nallocs == 0
    nallocs = @allocated push!(H, x, y)
    @test nallocs == 0
    nallocs = @allocated push!(C, x, y)
    @test nallocs == 0
    nallocs = @allocated push!(CD, x, y)
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
    C = CompactLBFGSOperator(n, mem = mem)
    for _ = 1:2:n
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
      push!(H, s, y)
      push!(C, s, y)
    end
    λ = eigs(B, nev = n)[1]  # call Arpack
    @test minimum(λ) > 0
    λ = eigs(H)[1]
    @test minimum(λ) > 0
    λ = eigs(C, nev = n)[1]
    @test minimum(λ) > 0
  end

  @testset ExtendedTestSet "Compact LBFGS vs LBFGS" begin
    n = 10
    rtol = sqrt(eps(Float64))

    # Same keyword arguments and defaults.
    @test CompactLBFGSOperator(n).data._mem == LBFGSOperator(n).data.mem
    @test CompactLBFGSOperator(n, mem = 0).data._mem == LBFGSOperator(n, mem = 0).data.mem

    for mem in (1, 3, n), scaling in (true, false), damped in (false, true)
      σs = damped ? ((0.99, 10.0), (0.8, Inf)) : ((0.99, 10.0),)
      for (σ₂, σ₃) in σs
        kwargs = (mem = mem, scaling = scaling, damped = damped, σ₂ = σ₂, σ₃ = σ₃)
        B = LBFGSOperator(n; kwargs...)
        C = CompactLBFGSOperator(n; kwargs...)
        for _ = 1:(3 * mem)
          # About half of the pairs have nonpositive curvature:
          # they are skipped, or damped when damped = true.
          s = randn(n)
          y = randn(n)
          push!(B, s, y)
          push!(C, s, y)
          @test norm(Matrix(C) - Matrix(B)) <= rtol * norm(Matrix(B))
        end

        # The compact factors give the same matrix: B = ξI - UUᵀ + VVᵀ.
        k = min(C.data._insert - 1, C.data._mem)
        U = C.data.Uk[:, 1:k]
        V = C.data.Vk[:, 1:k]
        Bk = C.data.ξ * Matrix(1.0I, n, n) - U * U' + V * V'
        @test norm(Bk - Matrix(B)) <= rtol * norm(Matrix(B))
      end
    end
  end

  @testset ExtendedTestSet "Compact LBFGS specifics" begin
    n = 10

    # Storage type
    @test LinearOperators.storage_type(CompactLBFGSOperator(n)) == Vector{Float64}
    C = CompactLBFGSOperator(Float32, n; S = Vector{Float32})
    @test LinearOperators.storage_type(C) == Vector{Float32}
    @test C.data.Sk isa Matrix{Float32}
    s = simple_vector(Float64, n)

    # The approximation is reset after more than max_skip consecutive skipped updates.
    C = CompactLBFGSOperator(n, max_skip = 1)
    push!(C, ones(n), ones(n))
    push!(C, s, -s)
    @test C.data._insert == 2
    push!(C, s, -s)
    @test C.data._insert == 1

    # With β = 0, the output vector is overwritten.
    push!(C, ones(n), 2 * ones(n))
    res = fill(NaN, n)
    mul!(res, C, s, 2.0, 0.0)
    @test res ≈ 2 * Matrix(C) * s

    # Copying API
    D = copy(C)
    @test Matrix(D) == Matrix(C)
    push!(D, s, 3 * s .+ 1)
    @test Matrix(D) != Matrix(C)
    E = similar(C)
    copy!(E, C)
    @test Matrix(E) == Matrix(C)
  end
end

test_lbfgs()
