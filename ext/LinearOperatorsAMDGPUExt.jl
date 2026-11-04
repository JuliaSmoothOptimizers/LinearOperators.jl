module LinearOperatorsAMDGPUExt

using LinearOperators
isdefined(Base, :get_extension) ? (using AMDGPU) : (using ..AMDGPU)

LinearOperators.storage_type(::ROCArray{T, 2, B}) where {T, B} = ROCArray{T, 1, B}

# Compact L-BFGS: work on the full arrays instead of views on the stored pairs.
LinearOperators._active(v::ROCVector, k) = v
LinearOperators._active(A::ROCMatrix, k) = A
LinearOperators._active(A::ROCMatrix, k, l) = A

end # module
