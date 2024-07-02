if Sys.isapple() && occursin("arm64", Sys.MACHINE)
  test_S_kwarg(arrayType = MtlArray, notMetal = false)
end