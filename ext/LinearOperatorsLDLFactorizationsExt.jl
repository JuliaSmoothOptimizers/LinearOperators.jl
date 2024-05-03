module LinearOperatorsLDLFactorizationsExt

using FastClosures, LDLFactorizations, LinearAlgebra, LinearOperators, SparseArrays

function LinearOperators.opLDL(M::AbstractMatrix; check::Bool = false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldlt(M)
  prod! = @closure (res, v, α, β) -> LinearOperators.mulFact!(res, LDL, v, α, β)
  tprod! = @closure (res, u, α, β) -> LinearOperators.tmulFact!(res, LDL, u, α, β)  # M.' = conj(M)
  ctprod! = @closure (res, w, α, β) -> LinearOperators.mulFact!(res, LDL, w, α, β)
  S = eltype(LDL)
  return LinearOperator{S}(m, m, isreal(M), true, prod!, tprod!, ctprod!)
  #TODO: use iterative refinement.
end

function LinearOperators.opLDL(
  M::Symmetric{T, SparseMatrixCSC{T, Int}};
  check::Bool = false,
) where {T <: Real}
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldl(M)
  prod! = @closure (res, v) -> ldiv!(res, LDL, v)
  tprod! = @closure (res, u) -> ldiv!(res, LDL, u)  # M.' = conj(M)
  ctprod! = @closure (res, w) -> ldiv!(res, LDL, w)
  S = eltype(LDL)
  return LinearOperator{S}(m, m, isreal(M), true, prod!, tprod!, ctprod!)
end

end # module
