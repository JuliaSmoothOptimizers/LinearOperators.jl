module LinearOperatorsAMDGPUExt

using LinearOperators
isdefined(Base, :get_extension) ? (using AMDGPU) : (using ..AMDGPU)

LinearOperators.storage_type(::ROCArray{T, 2, B}) where {T, B} = ROCArray{T, 1, B}

end # module
