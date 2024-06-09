function test_S_kwarg(; arrayType = JLArray, notMetal = true)
  mat = arrayType(rand(Float32, 32, 32))
  vec = arrayType(rand(Float32, 32))
  vecT = typeof(vec)

  if notMetal
    # To test operators which can derive a default storage_type from their arguments
    vecTother = typeof(arrayType(rand(Float32, 32)))
  end

  @testset ExtendedTestSet "S Kwarg with arrayType $(arrayType)" begin
    @test vecT == LinearOperators.storage_type(mat)

    # constructors.jl
    @test LinearOperators.storage_type(LinearOperator(mat)) == LinearOperators.storage_type(mat) # default
    notMetal && @test LinearOperators.storage_type(LinearOperator(mat; S = vecTother)) == vecTother
    @test LinearOperators.storage_type(LinearOperator(Symmetric(mat); S = vecT)) == vecT
    notMetal && @test LinearOperators.storage_type(LinearOperator(SymTridiagonal(Symmetric(mat)); S = vecT)) == vecT
    @test LinearOperators.storage_type(LinearOperator(Hermitian(mat); S = vecT)) == vecT
    @test LinearOperators.storage_type(LinearOperator(Float32, 32, 32, true, true, () -> 0; S = vecT)) == vecT

    # special-operators.jl
    @test LinearOperators.storage_type(opEye(Float32, 32; S = vecT)) == vecT
    @test LinearOperators.storage_type(opEye(Float32, 16, 32; S = vecT)) == vecT
    @test LinearOperators.storage_type(opEye(Float32, 32, 32; S = vecT)) == vecT

    @test LinearOperators.storage_type(opOnes(Float32, 32, 32; S = vecT)) == vecT
    @test LinearOperators.storage_type(opZeros(Float32, 32, 32; S = vecT)) == vecT

    @test LinearOperators.storage_type(opDiagonal(vec)) == vecT
    @test LinearOperators.storage_type(opDiagonal(32, 32, vec)) == vecT

    @test LinearOperators.storage_type(opRestriction([1, 2, 3], 32; S = vecT)) == vecT
    @test LinearOperators.storage_type(opExtension([1, 2, 3], 32; S = vecT)) == vecT

    @test LinearOperators.storage_type(BlockDiagonalOperator(mat, mat)) == vecT # default
    notMetal && @test LinearOperators.storage_type(BlockDiagonalOperator(mat, mat; S = vecTother)) == vecTother
  end

end

test_S_kwarg()
if Sys.isapple() && occursin("arm64", Sys.MACHINE)
    test_S_kwarg(arrayType = MtlArray, notMetal = false)
end