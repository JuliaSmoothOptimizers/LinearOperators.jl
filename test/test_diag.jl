# Points
x0 = [-1.0, 1.0, -1.0]
x1 = x0 + [1.0, 1.0, 1.0]

# Test functions
# f(x) = x[1]^2 + x[2]^2 + x[3]^2
∇f(x) = 2 * [x[1], x[2], x[3]]

# g(x) = exp(x[1]) + x[2] + cos(x[3])
∇g(x) = [exp(x[1]), 1, -sin(x[3])]

# h(x) = x[1]^2 * x[2] * x[3]^3
∇h(x) = [2 * x[1] * x[2] * x[3]^3, x[1]^2 * x[3]^3, 3 * x[1]^2 * x[2] * x[3]^2]

@testset "Weak secant equation" begin
  for grad_fun in (:∇f, :∇g, ∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    B = DiagonalQN([1.0, -1.0, 1.0])
    push!(B, s, y)
    @test abs(dot(s, B * s) - dot(s, y)) <= 1e-10
  end
end

@testset "Hard coded test" begin
  for grad_fun in (:∇f, :∇g, :∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    B = DiagonalQN([1.0, -1.0, 1.0])
    if grad_fun == :∇f
      Bref = [8 / 3, 8 / 3 - 2, 8 / 3]
    elseif grad_fun == :∇g
      Bref =
        [1 + (sin(-1) - exp(-1)) / 3, -1 + (sin(-1) - exp(-1)) / 3, 1 + (sin(-1) - exp(-1)) / 3]
    else
      Bref = [-2 / 3, -2 / 3 - 2, -2 / 3]
    end
    push!(B, s, y)
    @test norm(B.d - Bref) <= 1e-10

    B = SpectralGradient(1.0, 3)
    if grad_fun == :∇f
      Bref = 2
    elseif grad_fun == :∇g
      Bref = 1 / 3 * (1 - exp(-1) + sin(-1))
    else
      Bref = -4 / 3
    end
    push!(B, s, y)
    @test abs(B.d - Bref) <= 1e-10
  end
end
