module LinearOperatorsMetalExt

using LinearOperators
isdefined(Base, :get_extension) ? (using Metal) : (using ..Metal)

LinearOperators.storage_type(::MtlArray{T, 2, S}) where {T, S} = MtlArray{T, 1, S}

# Compact L-BFGS: work on the full arrays instead of views on the stored pairs.
LinearOperators._active(v::MtlVector, k) = v
LinearOperators._active(A::MtlMatrix, k) = A
LinearOperators._active(A::MtlMatrix, k, l) = A

end # module
