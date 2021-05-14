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
  function prod!(res, x, α, β)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), q, n)
    res .= Matrix(B * X * transpose(A))[:]
  end
  function tprod!(res, x, α, β)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    res .= Matrix(transpose(B) * X * A)[:]
  end
  function ctprod!(res, x, α, β)
    S = promote_type(T, eltype(x))
    X = reshape(convert(Vector{S}, x), p, m)
    res .= Matrix(B' * X * conj(A))[:]
  end
  symm = issymmetric(A) && issymmetric(B)
  herm = ishermitian(A) && ishermitian(B)
  nrow, ncol = m * p, n * q
  Mv = Vector{T}(undef, nrow)
  Mtu = symm ? Mv : Vector{T}(undef, ncol)
  Maw = herm ? Mv : Vector{T}(undef, ncol)
  return LinearOperator{T}(nrow, ncol, symm, herm, prod!, tprod!, ctprod!, Mv, Mtu, Maw)
end

kron(A::AbstractMatrix, B::AbstractLinearOperator) = kron(LinearOperator(A), B)

kron(A::AbstractLinearOperator, B::AbstractMatrix) = kron(A, LinearOperator(B))
