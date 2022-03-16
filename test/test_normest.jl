function test_normest()
  @testset "normest" begin
    ϵ = 0.001
    simple_matrix_test(ϵ)
  end
end

function simple_matrix_test_helper(S, ϵ)
  val = opnorm(Matrix(S), 2)
  val_normest, _ = normest(S, 1.0e-4, 10000)
  if val == 0
      dev = 1
  else
      dev = abs(val)
  end
  @test abs(val_normest - val) / dev <= ϵ
end

function simple_matrix_test(ϵ)
  (nrow, ncol) = (10, 10)  
  A1 = simple_matrix(ComplexF64, nrow, ncol)
  simple_matrix_test_helper(A1, ϵ)
  LA1 = LinearOperator(A1)
  simple_matrix_test_helper(LA1, ϵ)

  A2 = simple_matrix(Float64, nrow, ncol)
  simple_matrix_test_helper(A2, ϵ)
  LA2 = LinearOperator(A2)
  simple_matrix_test_helper(LA2, ϵ)
end

test_normest()
