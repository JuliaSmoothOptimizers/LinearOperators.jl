function test_linop()
  @testset "linop" begin
    (nrow, ncol) = (10, 6)
    ϵ = eps(Float64)
    rtol = sqrt(ϵ)
    A1 = rand(nrow, ncol)
    opA1 = LinearOperator(A1)
    v = rand(ncol)
    res = zeros(nrow)

    # test mul3
    mul!(res, opA1, v) # build 
    opA1 = LinearOperator(A1)
    allocs = @allocated mul!(res, opA1, v)
    @test allocs == 0 
    res_true = zeros(nrow)
    mul!(res_true, A1, v)
    @test norm(res-res_true) ≤ sqrt(ϵ)

    # test mul5 
    α, β = 2.0, -3.0
    res = rand(nrow)
    res_true .= res
    opA1 = LinearOperator(A1)
    mul!(res, opA1, v, α, β)
    mul!(res, A1, v, α, β)
    res .= res_true
    allocs2 = @allocated mul!(res, opA1, v, α, β)
    allocs = @allocated mul!(res_true, A1, v, α, β)
    @test allocs2 == 0
    @test allocs == 0
    @test norm(res-res_true) ≤ sqrt(ϵ)

    # test prod 
    opA1 = LinearOperator(A1)
    res = opA1 * v 
    res_true = A1 * v 
    @test norm(res-res_true) ≤ sqrt(ϵ)

    # test +op 
    opA2 = +opA1
    α, β = 2.0, -3.0
    res = rand(nrow)
    res_true .= res
    mul!(res, opA2, v, α, β)
    mul!(res_true, A1, v, α, β)
    @test norm(res-res_true) ≤ sqrt(ϵ)

    # test -op
    opA2 = -opA1
    α, β = 2.0, -3.0
    res = rand(nrow)
    res_true .= res
    opA1 = LinearOperator(A1)
    mul!(res, opA2, v, α, β)
    mul!(res_true, -A1, v, α, β)
    @test norm(res-res_true) ≤ sqrt(ϵ)
    mul!(res, opA2, v, α, β)

    # Matrix * LinearOperator
    (nrow2, ncol2) = (ncol, 7)
    v2 = rand(ncol2)
    A2 = rand(nrow2, ncol2)
    opA3 = opA1 * A2 
    Mv = rand(nrow)
    res_true = copy(Mv)
    mul!(Mv, opA3, v2, α, β)
    mul!(res_true, A1*A2, v2, α, β)
    @test norm(Mv-res_true) ≤ sqrt(ϵ)
  end
end

test_linop()
