module LinearOperatorsCUDAExt

using LinearOperators
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)

LinearOperators.storage_type(::CuArray{T, 2, D}) where {T, D} = CuArray{T, 1, D}

end # module
