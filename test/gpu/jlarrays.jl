test_S_kwarg(arrayType = JLArray)

JLArrays.allowscalar(false)
for T in (Float32, Float64)
  test_compact_lbfgs_gpu(arrayType = JLArray, T = T)
end
