module LinearOperatorsMetalExt

using LinearOperators
isdefined(Base, :get_extension) ? (using Metal) : (using ..Metal)

LinearOperators.storage_type(::MtlArray{T, 2, S}) where {T, S} = MtlArray{T, 1, S}

end # module