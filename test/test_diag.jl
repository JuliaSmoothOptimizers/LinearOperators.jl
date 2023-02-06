# Points
x0 = [-1.0, 1.0, -1.0]
x1 = x0 + [1.0, 0.0, 1.0]

# Test functions
# f(x) = x[1]^2 + x[2]^2 + x[3]^2
∇f(x) = 2 * [x[1], x[2], x[3]]

# g(x) = exp(x[1]) + x[2] + cos(x[3])
∇g(x) = [exp(x[1]), 1, -sin(x[3])]

# h(x) = x[1]^2 * x[2] * x[3]^3
∇h(x) = [2 * x[1] * x[2] * x[3]^3, x[1]^2 * x[3]^3, 3 * x[1]^2 * x[2] * x[3]^2]

@testset "Weak secant equation for Andrei update" begin
  for grad_fun in (:∇f, :∇g, ∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    B = DiagonalQN([1.0, -1.0, 1.0])
    push!(B, s, y)
    @test abs(dot(s, B * s) - dot(s, y)) <= 1e-10
  end
end

@testset "Weak secant equation for PSB update" begin
  for grad_fun in (:∇f, :∇g, ∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    B = DiagonalQN([1.0, -1.0, 1.0], true)
    push!(B, s, y)
    @test abs(dot(s, B * s) - dot(s, y)) <= 1e-10
  end
end

@testset "Hard coded test" begin
  for grad_fun in (:∇f, :∇g, :∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    for psb ∈ (false, true)
      B = DiagonalQN([1.0, -1.0, 1.0], psb)
      if grad_fun == :∇f
        Bref = psb ? [2, -1, 2] : [2, -2, 2]
      elseif grad_fun == :∇g
        Bref =
          psb ? [1 + (sin(-1) - exp(-1) - 1) / 2, -1, 1 + (sin(-1) - exp(-1) - 1) / 2] :
          [(1 + sin(-1) - exp(-1)) / 2, -2, (1 + sin(-1) - exp(-1)) / 2]
      else
        Bref = psb ? [-5 / 2, -1, -5 / 2] : [-5 / 2, -2, -5 / 2]
      end
      push!(B, s, y)
      @test norm(B.d - Bref) <= 1e-10
    end

    B = SpectralGradient(1.0, 3)
    if grad_fun == :∇f
      Bref = 2
    elseif grad_fun == :∇g
      Bref = (1 - exp(-1) + sin(-1)) / 2
    else
      Bref = -5 / 2
    end
    push!(B, s, y)
    @test abs(B.d[1] - Bref) <= 1e-10
  end
end

@testset "Allocations test" begin
  d = rand(5)
  A = DiagonalQN(d)
  v = rand(5)
  u = similar(v)
  mul!(u, A, v)
  @test (@allocated mul!(u, A, v)) == 0
  B = DiagonalQN(d, true)
  mul!(u, B, v)
  @test (@allocated mul!(u, B, v)) == 0
  C = SpectralGradient(rand(), 5)
  mul!(u, C, v)
  @test (@allocated mul!(u, C, v)) == 0
end

@testset "reset" begin
  B = DiagonalQN([1.0, -1.0, 1.0], false)
  s = x1 - x0
  y = ∇f(x1) - ∇f(x0)
  push!(B, s, y)
  reset!(B)
  @test B * x0 == x0

  B = SpectralGradient(2.5, 3)
  s = x1 - x0
  y = ∇f(x1) - ∇f(x0)
  push!(B, s, y)
  reset!(B)
  @test B * x0 == x0
end
