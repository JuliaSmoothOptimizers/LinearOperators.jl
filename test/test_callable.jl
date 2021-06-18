struct Flip end

function (::Flip)(res, x, α, β::T) where {T}
  if β == zero(T)
    res .= (-α) .* x
  else
    res .= (-α) .* x .+ β .* res
  end
end

function test_callable()
  @testset ExtendedTestSet "Test callable" begin
    Mv = ones(2)
    op = LinearOperator(Float64, 2, 2, true, true, Flip())
    @test op * ones(2) == -ones(2)
    @test op' * ones(2) == -ones(2)
    @test transpose(op) * ones(2) == -ones(2)
    v = ones(2)
    allocs = @allocated mul!(Mv, op, v)
    @test allocs == 0
    @test Mv == -ones(2) 
  end
end

test_callable()
