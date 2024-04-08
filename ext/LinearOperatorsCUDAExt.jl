module LinearOperatorsCUDAExt

using LinearOperators, LinearOperators.FastClosures, LinearOperators.LinearAlgebra
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)

function LinearOperators.LinearOperator(
  M::CuArray{T, 2, D};
  symmetric = false,
  hermitian = false,
  S = CuArray{T, 1, D},
) where {T, D}
  nrow, ncol = size(M)
  prod! = @closure (res, v, α, β) -> mul!(res, M, v, α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(M), u, α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(M), w, α, β)
  LinearOperators.LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, S = S)
end

end # module
