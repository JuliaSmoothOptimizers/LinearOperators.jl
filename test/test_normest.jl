function test_normest()
  @testset "normest" begin
    ϵ = 0.01
    simple_matrix_test(ϵ)
  end
end

function simple_matrix_test_helper(S, ϵ)
  val = opnorm(S, 2)
  val_normest, _ = normest(S, 1.0e-4, 10000)
  if val == 0
      dev = 1
  else
      dev = abs(val)
  end
  @test abs(val_normest - val) / dev <= ϵ
end

function simple_matrix_test(ϵ)
  S = reshape(collect(1:16), (4, 4))
  simple_matrix_test_helper(S, ϵ)
  S = reshape(collect(1:160), (40, 4))
  simple_matrix_test_helper(S, ϵ)
  S = reshape(collect(1:400), (20, 20))
  simple_matrix_test_helper(S, ϵ)

end
#running the tests
test_normest()
