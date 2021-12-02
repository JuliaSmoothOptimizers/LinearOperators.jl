function test_deprecated()
  @testset ExtendedTestSet "Deprecated methods" begin
    n = 10
    mem = 3
    T = Float16
    @test_deprecated LBFGSOperator(n, mem, scaling = false)
    @test_deprecated InverseLBFGSOperator(n, mem, scaling = false)
    @test_deprecated LBFGSOperator(T, n, mem, scaling = false)
    @test_deprecated InverseLBFGSOperator(T, n, mem, scaling = false)
    @test_deprecated LSR1Operator(n, mem, scaling = false)
    @test_deprecated LSR1Operator(T, n, mem, scaling = false)
    opA = LinearOperator(rand(10,10))
    for op in [opA, TimedLinearOperator(opA), adjoint(opA), transpose(opA), conj(opA)]
      @test_deprecated hermitian(op)
      @test_deprecated symmetric(op)
    end
  end
end

test_deprecated()
