using Test, LinearAlgebra, LinearOperators

function test_lbfgs_backend(; arrayType)
  T = Float32
  n = 8
  x = T.(1:n) ./ n
  v = arrayType(x)
  V = typeof(v)
  rtol = 5.0f-4
  atol = 5.0f-5

  @testset "LBFGS with $arrayType" begin
    @testset "mem=$mem, scaling=$scaling" for mem in (0, 3), scaling in (false, true)
      B = LBFGSOperator(T, V, n; mem = mem, scaling = scaling)
      H = InverseLBFGSOperator(T, V, n; mem = mem, scaling = scaling)
      maxmem = max(mem, 1)
      for op in (B, H)
        @test LinearOperators.storage_type(op) == V
        @test op.data.mem == maxmem
        @test all(w -> w isa V, op.data.s)
        @test all(w -> w isa V, op.data.y)
        @test op.data.Ax isa V
        @test op.data.shifted_p isa typeof(similar(v, n, 2 * maxmem))
        @test size(op.data.shifted_p) == (n, 2 * maxmem)
        @test op * v isa V
        @test Array(op * v) ≈ x
        # Rejected curvature must not change the approximation.
        push!(op, v, -v)
        push!(op, v, zero(v))
        @test op.data.insert == 1
        @test Array(op * v) ≈ x
      end

      pairs = Tuple{Vector{T}, Vector{T}}[]
      # Check empty, partially filled, full, and wrapped history.
      for k = 0:(maxmem + 2)
        if k > 0
          s = T[sin(i + k) for i = 1:n]
          y = (one(T) .+ T.(1:n) ./ n) .* s
          push!(pairs, (s, y))
          length(pairs) > maxmem && popfirst!(pairs)
          push!(B, arrayType(s), arrayType(y))
          push!(H, arrayType(s), arrayType(y))
        end
        # Independent dense BFGS reference using the retained history.
        γ = scaling && k > 0 ? dot(last(pairs)...) / dot(last(pairs)[2], last(pairs)[2]) : one(T)
        dense = Matrix{T}(I, n, n) / γ
        for (s, y) in pairs
          Bs = dense * s
          dense += y * y' / dot(s, y) - Bs * Bs' / dot(s, Bs)
        end
        for (op, expected) in ((B, dense * x), (H, dense \ x))
          @test op.data.insert == mod(k, maxmem) + 1
          @test Array(op * v) ≈ expected rtol = rtol atol = atol
          out = copy(v)
          mul!(out, op, v, T(2), T(0.5))
          @test Array(out) ≈ 2 .* expected .+ T(0.5) .* x rtol = rtol atol = atol
          fill!(out, T(NaN))
          mul!(out, op, v, one(T), zero(T))
          @test Array(out) ≈ expected rtol = rtol atol = atol
        end
        @test diag(B) isa V
        @test Array(diag(B)) ≈ diag(dense) rtol = rtol atol = atol
        d = similar(v)
        @test diag!(B, d) === d
        @test Array(d) ≈ diag(dense) rtol = rtol atol = atol
        sol = similar(v)
        for σ in (zero(T), T(0.5))
          @test solve_shifted_system!(sol, B, v, σ) === sol
          @test Array(sol) ≈ (dense + σ * I) \ x rtol = rtol atol = atol
          @test Array(B * sol + σ * sol) ≈ x rtol = rtol atol = atol
        end
        @test ldiv!(sol, B, v) === sol
        @test Array(sol) ≈ dense \ x rtol = rtol atol = atol
      end
      @test_throws LinearOperators.LinearOperatorException diag(H)
      @test_throws ArgumentError solve_shifted_system!(similar(v), B, v, -one(T))
      for op in (B, H)
        reset!(op)
        @test op.data.insert == 1
        @test op.data.scaling_factor == one(T)
        @test Array(op * v) ≈ x
      end
      @test Array(diag(B)) ≈ ones(T, n)
    end

    @testset "damped updates" for constructor in (LBFGSOperator, InverseLBFGSOperator)
      op = constructor(T, V, n; mem = 3, damped = true)
      cpu = constructor(T, n; mem = 3, damped = true)
      for k = 1:5
        g = T[cos(i + k) for i = 1:n]
        s = -(cpu * g)
        # Exercise both lower and upper curvature damping.
        y = (isodd(k) ? T(0.001) : T(100)) .* (-g)
        if op.inverse
          if isodd(k)
            push!(op, arrayType(s), arrayType(y), one(T), arrayType(g))
            push!(cpu, s, copy(y), one(T), g)
          else
            push!(op, arrayType(s), arrayType(y), one(T), arrayType(g), similar(v))
            push!(cpu, s, copy(y), one(T), g, similar(x))
          end
        elseif isodd(k)
          push!(op, arrayType(s), arrayType(y))
          push!(cpu, s, y)
        else
          push!(op, arrayType(s), arrayType(y), similar(v))
          push!(cpu, s, y, similar(x))
        end
        @test Array(op * v) ≈ cpu * x rtol = rtol atol = atol
        @test op.data.insert == cpu.data.insert
      end
    end
  end
end
