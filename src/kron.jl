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
  function prod(x)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), q, n)
    return Matrix(B * X * transpose(A))[:]
  end
  function tprod(x)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    return Matrix(transpose(B) * X * A)[:]
  end
  function ctprod(x)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    return Matrix(B' * X * conj(A))[:]
  end
  symm = issymmetric(A) && issymmetric(B)
  herm = ishermitian(A) && ishermitian(B)
  return LinearOperator{T}(m * p, n * q, symm, herm, prod, tprod, ctprod)
end

kron(A::AbstractMatrix, B::AbstractLinearOperator) = kron(LinearOperator(A), B)

kron(A::AbstractLinearOperator, B::AbstractMatrix) = kron(A, LinearOperator(B))
