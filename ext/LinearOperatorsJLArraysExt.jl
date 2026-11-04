module LinearOperatorsJLArraysExt

using LinearOperators
isdefined(Base, :get_extension) ? (using JLArrays) : (using ..JLArrays)

LinearOperators.storage_type(::JLArray{T, 2}) where {T} = JLArray{T, 1}

# Compact L-BFGS: work on the full arrays instead of views on the stored pairs.
LinearOperators._active(v::JLVector, k) = v
LinearOperators._active(A::JLMatrix, k) = A
LinearOperators._active(A::JLMatrix, k, l) = A

end # module
