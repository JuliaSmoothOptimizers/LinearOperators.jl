using Test
using LinearOperators
using LinearAlgebra

function setup_test_val(; M = 5, n = 100, scaling = false, σ = 0.1)
  B = LBFGSOperator(n, mem = M, scaling = scaling)
  H = InverseLBFGSOperator(n, mem = M, scaling = false)

  for _ = 1:10
    s = rand(n)
    y = rand(n)
    push!(B, s, y)
    push!(H, s, y)
  end

  x = randn(n)
  b = B * x + σ .* x # so we know the true answer is x

  return B, H , b, σ, zeros(n), x
end

function test_solve_shifted_system()
  @testset "solve_shifted_system! Default setup test" begin
    # Setup Test Case 1: Default setup from setup_test_val
    B,_, b, σ, x_sol, x_true = setup_test_val(n = 100, M = 5)

    result = solve_shifted_system!(x_sol, B, b, σ)

    # Test 1: Check if result is a vector of the same size as z
    @test length(result) == length(b)

    # Test 2: Verify that x_sol (result) is modified in place
    @test result === x_sol

    # Test 3: Check if the function produces finite values
    @test all(isfinite, result)

    # Test 4: Check if x_sol is close to the known solution x
    @test isapprox(x_sol, x_true, atol = 1e-6, rtol = 1e-6)
  end
  @testset "solve_shifted_system! Negative σ test" begin
    # Setup Test Case 2: Negative σ
    B,_, b, _, x_sol, _ = setup_test_val(n = 100, M = 5)
    σ = -0.1

    # Expect an ArgumentError to be thrown
    @test_throws ArgumentError solve_shifted_system!(x_sol, B, b, σ)
  end

  @testset "ldiv! test" begin
    # Setup Test Case 1: Default setup from setup_test_val
    B, H, b, _, x_sol, x_true = setup_test_val(n = 100, M = 5, σ = 0.0)

    # Solve the system using solve_shifted_system!
    result = ldiv!(x_sol, B, b)

    # Check consistency with operator-vector product using H
    x_H = H * b
    @test isapprox(x_sol, x_H, atol = 1e-6, rtol = 1e-6)
    @test isapprox(x_sol, x_true, atol = 1e-6, rtol = 1e-6)
  end
end

test_solve_shifted_system()
