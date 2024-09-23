using Test
using LinearOperators
using LinearAlgebra

function setup_test_val(; M = 5, n = 100, scaling = false)
  σ = 0.1
  B = LBFGSOperator(n, mem = 5, scaling = scaling)
  s = ones(n)
  y = ones(n)

  S = randn(n, M)
  Y = randn(n, M)

  # make sure it is positive
  for i = 1:M
    if dot(S[:, i], Y[:, i]) < 0
      S[:, i] = -S[:, i]
    end
  end
  for i = 1:M
    s = S[:, i]
    y = Y[:, i]
    if dot(s, y) > 1.0e-20
      push!(B, s, y)
    end
  end

  γ_inv = 1 / B.data.scaling_factor

  x = randn(n)
  z = B * x + σ .* x # so we know the true answer is x

  data = B.data
  # Preallocate vectors for efficiency
  p = zeros(size(data.a[1], 1), 2 * (data.mem))
  v = zeros(2 * (data.mem))
  u = zeros(size(data.a[1], 1))

  return B, z, σ, γ_inv, zeros(n), p, v, u, x
end

function test_solve_shifted_system()
  @testset "solve_shifted_system! Default setup test" begin
    # Setup Test Case 1: Default setup from setup_test_val
    B, z, σ, γ_inv, inv_Cz, p, v, u, x = setup_test_val(n = 100, M = 5)

    result = solve_shifted_system!(B, z, σ, γ_inv, inv_Cz, p, v, u)

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
  @testset "solve_shifted_system!  Larger dimensional system tests" begin

    # Test Case 2: Larger dimensional system
    dim = 10
    mem_size = 5
    B, z, σ, γ_inv, inv_Cz, p, v, u, x = setup_test_val(n = dim, M = mem_size)

    result = solve_shifted_system!(B, z, σ, γ_inv, inv_Cz, p, v, u)

    # Test 5: Check if result is a vector of the same size as z (larger case)
    @test length(result) == length(z)

    # Test 6: Verify that inv_Cz is modified in place (larger case)
    @test result === inv_Cz

    # Test 7: Check if the function produces finite values (larger case)
    @test all(isfinite, result)

    # Test 8: Check if inv_Cz is close to the known solution x (larger case)
    # x = B \ (z ./ (1 + σ))
    @test isapprox(inv_Cz, x, atol = 1e-6, rtol = 1e-6)
  end

  @testset "solve_shifted_system! Minimal memory size test" begin
    # Test Case 3: Minimal memory size case (memory size = 1)
    dim = 4
    mem_size = 1
    B, z, σ, γ_inv, inv_Cz, p, v, u, x = setup_test_val(n = dim, M = mem_size)

    result = solve_shifted_system!(B, z, σ, γ_inv, inv_Cz, p, v, u)

    # Test 9: Check if result is a vector of the same size as z (minimal memory)
    @test length(result) == length(z)

    # Test 10: Verify that inv_Cz is modified in place (minimal case)
    @test result === inv_Cz

    # Test 11: Check if the function produces finite values (minimal case)
    @test all(isfinite, result)

    # Test 12: Check if inv_Cz is close to the known solution x (minimal memory)
    # x = B \ (z ./ (1 + σ))
    @test isapprox(inv_Cz, x, atol = 1e-6, rtol = 1e-6)
  end

  @testset "solve_shifted_system! Extra large memory size test" begin

    # Test Case 4: Even larger system with more memory (case 4)
    dim = 50
    mem_size = 10
    B, z, σ, γ_inv, inv_Cz, p, v, u, x = setup_test_val(n = dim, M = mem_size)

    # Call the function
    result = solve_shifted_system!(B, z, σ, γ_inv, inv_Cz, p, v, u)

    # Test 13: Check if result is a vector of the same size as z (case 4)
    @test length(result) == length(z)

    # Test 14: Verify that inv_Cz is modified in place (case 4)
    @test result === inv_Cz

    # Test 15: Check if the function produces finite values (case 4)
    @test all(isfinite, result)

    # Test 16: Check if inv_Cz is close to the known solution x (case 4)
    # x = B \ (z ./ (1 + σ))
    @test isapprox(inv_Cz, x, atol = 1e-6, rtol = 1e-6)
  end
end

test_solve_shifted_system()
