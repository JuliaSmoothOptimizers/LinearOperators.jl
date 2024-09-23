using Test
using LinearOperators
using LinearAlgebra

function setup_test_val(; M = 5, n = 100, scaling = false)
  σ = 0.1
  B = LBFGSOperator(n, mem = 5, scaling = scaling)

  for _ = 1:10
    s = rand(n)
    y = rand(n)
    push!(B, s, y)
  end

  # γ_inv = 1 / B.data.scaling_factor

  x = randn(n)
  z = B * x + σ .* x # so we know the true answer is x

  return B, z, σ, zeros(n), x
end

function test_solve_shifted_system()
  @testset "solve_shifted_system! Default setup test" begin
    # Setup Test Case 1: Default setup from setup_test_val
    B, z, σ, inv_Cz, x = setup_test_val(n = 100, M = 5)

    result = solve_shifted_system!(B, z, σ, inv_Cz)

    # Test 1: Check if result is a vector of the same size as z
    @test length(result) == length(z)

    # Test 2: Verify that inv_Cz (result) is modified in place
    @test result === inv_Cz

    # Test 3: Check if the function produces finite values
    @test all(isfinite, result)

    # Test 4: Check if inv_Cz is close to the known solution x
    # x = B \ (z ./ (1 + σ))  # Known true solution
    @test isapprox(inv_Cz, x, atol = 1e-6, rtol = 1e-6)
  end

end

test_solve_shifted_system()
