struct Flip
end

function (::Flip)(x)
  return -x
end

function test_callable()
  @testset "Test callable" begin
    op = LinearOperator(2, 2, true, true, Flip())
    @test op * ones(2) == -ones(2)
    @test op' * ones(2) == -ones(2)
    @test transpose(op) * ones(2) == -ones(2)
  end
end

test_callable()
