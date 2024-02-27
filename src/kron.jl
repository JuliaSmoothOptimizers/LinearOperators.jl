import Base.kron

# (A ⊗ B)×vec(X) = vec(BXAᵀ)
"""`kron(A, B)`

Kronecker tensor product of A and B in linear operator form, if either
or both are linear operators. If both A and B are matrices, then
`Base.kron` is used.
"""
function kron(A::AbstractLinearOperator, B::AbstractLinearOperator)
  m, n = size(A)
  p, q = size(B)
  T = promote_type(eltype(A), eltype(B))
  function prod!(res, x, α, β::T2) where {T2}
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), q, n)
    if β == zero(T2)
      res .= α .* Matrix(B * X * transpose(A))[:]
    else
      res .= α .* Matrix(B * X * transpose(A))[:] .+ β .* res
    end
  end
  function tprod!(res, x, α, β::T2) where {T2}
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    if β == zero(T2)
      res .= α .* Matrix(transpose(B) * X * A)[:]
    else
      res .= α .* Matrix(transpose(B) * X * A)[:] .+ β .* res
    end
  end
  function ctprod!(res, x, α, β::T2) where {T2}
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    if β == zero(T2)
      res .= α .* Matrix(B' * X * conj(A))[:]
    else
      res .= α .* Matrix(B' * X * conj(A))[:] .+ β .* res
    end
  end
  symm = issymmetric(A) && issymmetric(B)
  herm = ishermitian(A) && ishermitian(B)
  nrow, ncol = m * p, n * q
  return LinearOperator5(T, nrow, ncol, symm, herm, prod!, tprod!, ctprod!)
end

kron(A::AbstractMatrix, B::AbstractLinearOperator) = kron(LinearOperator(A), B)

kron(A::AbstractLinearOperator, B::AbstractMatrix) = kron(A, LinearOperator(B))
