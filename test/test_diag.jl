"""
    @wrappedallocs(expr)

Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).

This code is based on that of https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/master/test/runtests.jl

For example, `@wrappedallocs(x + y)` produces:

```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```

You can use this macro in a unit test to verify that a function does not
allocate:

```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

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
    B = DiagonalAndrei([1.0, -1.0, 1.0])
    push!(B, s, y)
    @test abs(dot(s, B * s) - dot(s, y)) <= 1e-10
  end
end

@testset "Weak secant equation for PSB update" begin
  for grad_fun in (:∇f, :∇g, ∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    B = DiagonalPSB([1.0, -1.0, 1.0])
    push!(B, s, y)
    @test abs(dot(s, B * s) - dot(s, y)) <= 1e-10
  end
end

@testset "Hard coded test" begin
  Bref = Dict{Symbol, Dict{Any, Any}}()
  Bref[:∇f] = Dict{Any, Any}()
  Bref[:∇g] = Dict{Any, Any}()
  Bref[:∇h] = Dict{Any, Any}()
  Bref[:∇f][DiagonalPSB] = [2, -1, 2]
  Bref[:∇f][DiagonalAndrei] = [2, -2, 2]
  Bref[:∇g][DiagonalPSB] = [1 + (sin(-1) - exp(-1) - 1) / 2, -1, 1 + (sin(-1) - exp(-1) - 1) / 2]
  Bref[:∇g][DiagonalAndrei] = [(1 + sin(-1) - exp(-1)) / 2, -2, (1 + sin(-1) - exp(-1)) / 2]
  Bref[:∇h][DiagonalPSB] = [-5 / 2, -1, -5 / 2]
  Bref[:∇h][DiagonalAndrei] = [-5 / 2, -2, -5 / 2]

  Bref_spg = Dict{Any, Any}()
  Bref_spg[:∇f] = 2
  Bref_spg[:∇g] = (1 - exp(-1) + sin(-1)) / 2
  Bref_spg[:∇h] = -5 / 2

  for grad_fun in (:∇f, :∇g, :∇h)
    grad = eval(grad_fun)
    s = x1 - x0
    y = grad(x1) - grad(x0)
    for DQN ∈ (DiagonalPSB, DiagonalAndrei)
      B = DQN([1.0, -1.0, 1.0])
      push!(B, s, y)
      @test norm(B.d - Bref[grad_fun][DQN]) <= 1e-10
    end

    B = SpectralGradient(1.0, 3)
    push!(B, s, y)
    @test abs(B.d[1] - Bref_spg[grad_fun]) <= 1e-10
  end
end

@testset "Allocations test" begin
  d = rand(5)
  A = DiagonalAndrei(d)
  v = rand(5)
  u = similar(v)
  mul!(u, A, v)
  @test (@allocated mul!(u, A, v)) == 0
  @test (@wrappedallocs push!(A, u, v)) == 0
  B = DiagonalPSB(d)
  mul!(u, B, v)
  @test (@allocated mul!(u, B, v)) == 0
  @test (@wrappedallocs push!(B, u, v)) == 0
  C = SpectralGradient(rand(), 5)
  mul!(u, C, v)
  @test (@allocated mul!(u, C, v)) == 0
  @test (@wrappedallocs push!(C, u, v)) == 0
  D = DiagonalBFGS(d)
  mul!(u, D, v)
  @test (@allocated mul!(u, D, v)) == 0
  @test (@wrappedallocs push!(D, u, v)) == 0
end

@testset "reset" begin
  B = DiagonalAndrei([1.0, -1.0, 1.0])
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
