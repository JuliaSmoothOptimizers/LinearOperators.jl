module LinearOperatorsCUDAExt

using LinearOperators
isdefined(Base, :get_extension) ? (using CUDA; using CUDA.cuSPARSE) :
(using ..CUDA; using ..CUDA.cuSPARSE)

LinearOperators.storage_type(::CuArray{T, 2, D}) where {T, D} = CuArray{T, 1, D}
LinearOperators.storage_type(::AbstractCuSparseMatrix{T}) where {T} =
  CuArray{T, 1, CUDA.DeviceMemory}

# Compact L-BFGS: work on the full arrays instead of views on the stored pairs.
LinearOperators._active(v::CuVector, k) = v
LinearOperators._active(A::CuMatrix, k) = A
LinearOperators._active(A::CuMatrix, k, l) = A

end # module
