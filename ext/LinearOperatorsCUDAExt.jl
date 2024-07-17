module LinearOperatorsCUDAExt

using LinearOperators
isdefined(Base, :get_extension) ? (using CUDA; using CUDA.CUSPARSE) : (using ..CUDA; using ..CUDA.CUSPARSE)

LinearOperators.storage_type(::CuArray{T, 2, D}) where {T, D} = CuArray{T, 1, D}
LinearOperators.storage_type(::AbstractCuSparseMatrix{T}) where {T} = CuArray{T, 1, CUDA.DeviceMemory}

end # module
