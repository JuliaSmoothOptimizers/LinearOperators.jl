function test_adjtrans()
  @testset ExtendedTestSet "Adjoint/Transpose/Conjugate" begin
    A = rand(5, 3) + im * rand(5, 3)
    opA = LinearOperator(A)

    aopA = AdjointLinearOperator(opA)
    copA = ConjugateLinearOperator(opA)
    topA = TransposeLinearOperator(opA)

    for (foo, fop) in [(adjoint, aopA), (conj, copA), (transpose, topA)]
      @test foo(opA) === fop
      @test Matrix(fop) == foo(A)
      @test foo(fop) === opA

      @test Matrix(-fop) == foo(-A)
      @test Matrix((2 + 3im) * fop) ≈ (2 + 3im) * foo(A)
      @test Matrix(fop * (2 + 3im)) ≈ foo(A) * (2 + 3im)
    end

    @test adjoint(topA) === copA
    @test adjoint(copA) === topA
    @test conj(aopA) === topA
    @test conj(topA) === aopA
    @test transpose(copA) === aopA
    @test transpose(aopA) === copA

    v = rand(5) + im * rand(5)
    @test aopA * v == adjoint(A) * v
    @test topA * v == transpose(A) * v
    v = rand(5)
    @test aopA * v ≈ adjoint(A) * v
    @test topA * v ≈ transpose(A) * v

    v = rand(3) + im * rand(3)
    @test copA * v ≈ conj(A) * v
    v = rand(3)
    @test copA * v ≈ conj(A) * v
  end
end

test_adjtrans()

##
function test_derived_adjoint()
  @testset ExtendedTestSet "Derived Adjoint" begin
    A = rand(5, 3) + im * rand(5, 3)

    opA = LinearOperator{eltype(A)}(
      size(A, 1),
      size(A, 2),
      false,
      false,
      (y, x) -> mul!(y, A, x),
      (x, y) -> mul!(x, transpose(A), y),
      nothing,
    )

    aopA = AdjointLinearOperator(opA)
    copA = ConjugateLinearOperator(opA)
    topA = TransposeLinearOperator(opA)

    for (foo, fop) in [(adjoint, aopA), (conj, copA), (transpose, topA)]
      @test foo(opA) === fop
      @test Matrix(fop) == foo(A)
      @test foo(fop) === opA

      @test Matrix(-fop) == foo(-A)
      @test Matrix((2 + 3im) * fop) ≈ (2 + 3im) * foo(A)
      @test Matrix(fop * (2 + 3im)) ≈ foo(A) * (2 + 3im)
    end

    @test adjoint(topA) === copA
    @test adjoint(copA) === topA
    @test conj(aopA) === topA
    @test conj(topA) === aopA
    @test transpose(copA) === aopA
    @test transpose(aopA) === copA

    v = rand(5) + im * rand(5)
    @test aopA * v == adjoint(A) * v
    @test topA * v == transpose(A) * v
    v = rand(5)
    @test aopA * v ≈ adjoint(A) * v
    @test topA * v ≈ transpose(A) * v

    v = rand(3) + im * rand(3)
    @test copA * v ≈ conj(A) * v
    v = rand(3)
    @test copA * v ≈ conj(A) * v
  end
end

test_derived_adjoint()

##
function test_derived_transpose()
  @testset ExtendedTestSet "Derived Transpose" begin
    A = rand(5, 3) + im * rand(5, 3)

    opA = LinearOperator{eltype(A)}(
      size(A, 1),
      size(A, 2),
      false,
      false,
      (y, x) -> mul!(y, A, x),
      nothing,
      (x, y) -> mul!(x, adjoint(A), y),
    )

    aopA = AdjointLinearOperator(opA)
    copA = ConjugateLinearOperator(opA)
    topA = TransposeLinearOperator(opA)

    for (foo, fop) in [(adjoint, aopA), (conj, copA), (transpose, topA)]
      @test foo(opA) === fop
      @test Matrix(fop) == foo(A)
      @test foo(fop) === opA

      @test Matrix(-fop) == foo(-A)
      @test Matrix((2 + 3im) * fop) ≈ (2 + 3im) * foo(A)
      @test Matrix(fop * (2 + 3im)) ≈ foo(A) * (2 + 3im)
    end

    @test adjoint(topA) === copA
    @test adjoint(copA) === topA
    @test conj(aopA) === topA
    @test conj(topA) === aopA
    @test transpose(copA) === aopA
    @test transpose(aopA) === copA

    v = rand(5) + im * rand(5)
    @test aopA * v == adjoint(A) * v
    @test topA * v == transpose(A) * v
    v = rand(5)
    @test aopA * v ≈ adjoint(A) * v
    @test topA * v ≈ transpose(A) * v

    v = rand(3) + im * rand(3)
    @test copA * v ≈ conj(A) * v
    v = rand(3)
    @test copA * v ≈ conj(A) * v
  end
end

test_derived_transpose()
