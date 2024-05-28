module LinearOperatorsJLArraysExt

using LinearOperators
isdefined(Base, :get_extension) ? (using JLArrays) : (using ..JLArrays)

LinearOperators.storage_type(::JLArray{T, 2}) where {T} = JLArray{T, 1}

end # module
