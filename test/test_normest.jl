function test_normest()
  @testset "normest" begin
    ϵ = 0.001
    simple_matrix_test(ϵ)
  end
end

function simple_matrix_test_helper(S, ϵ_norm, ϵ)
  val = opnorm(Matrix(S), 2)
  val_normest, _ = normest(S, ϵ_norm, 10000)
  if val == 0
    dev = 1
  else
    dev = abs(val)
  end
  @test abs(val_normest - val) / dev <= ϵ
end

function simple_matrix_test(ϵ)
  (nrow, ncol) = (10, 10)
  ϵ_norm = eps(Float64)
  for (nrow, ncol) in [(10, 10), (5, 5), (3, 3)]
    
    A1 = simple_matrix(ComplexF64, nrow, ncol)
    simple_matrix_test_helper(A1, ϵ_norm, ϵ)
    LA1 = LinearOperator(A1)
    simple_matrix_test_helper(LA1, ϵ_norm, ϵ)

    A2 = simple_matrix(Float64, nrow, ncol)
    simple_matrix_test_helper(A2, ϵ_norm, ϵ)
    LA2 = LinearOperator(A2)
    ϵ_norm = -1
    simple_matrix_test_helper(LA2, ϵ_norm, ϵ)

    A3 = zeros(Float64, nrow, ncol)
    simple_matrix_test_helper(A3, ϵ_norm, ϵ)
    LA3 = LinearOperator(A3)
    ϵ_norm = -1
    simple_matrix_test_helper(LA3, ϵ_norm, ϵ)
  end
end

test_normest()
