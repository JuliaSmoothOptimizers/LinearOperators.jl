export opInverse, opCholesky, opLDL, opHouseholder, opHermitian

"""
    opInverse(M; symmetric=false, hermitian=false)

Inverse of a matrix as a linear operator using `\\`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`.
"""
function opInverse(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) where T
  prod = @closure v -> M \ v
  tprod = @closure u -> transpose(M) \ u
  ctprod = @closure w -> M' \ w
  LinearOperator{T}(size(M,2), size(M,1), symmetric, hermitian, prod, tprod, ctprod)
end

"""
    opCholesky(M, [check=false])

Inverse of a Hermitian and positive definite matrix as a linear operator
using its Cholesky factorization. The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks.
"""
function opCholesky(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
    check_positive_definite(M) || throw(LinearOperatorException("matrix is not positive definite"))
  end
  LL = cholesky(M)
  prod = @closure v -> LL \ v
  tprod = @closure u -> conj(LL \ conj(u))  # M.' = conj(M)
  ctprod = @closure w -> LL \ w
  S = eltype(LL)
  LinearOperator{S}(m, m, isreal(M), true, prod, tprod, ctprod)
  #TODO: use iterative refinement.
end

"""
    opLDL(M, [check=false])

Inverse of a symmetric matrix as a linear operator using its LDL' factorization
if it exists. The factorization is computed only once. The optional `check`
argument will perform a cheap hermicity check.
"""
function opLDL(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldlt(M)
  prod = @closure v -> LDL \ v
  tprod = @closure u -> conj(LDL \ conj(u))  # M.' = conj(M)
  ctprod = @closure w -> LDL \ w
  S = eltype(LDL)
  return LinearOperator{S}(m, m, isreal(M), true, prod, tprod, ctprod)
  #TODO: use iterative refinement.
end

"""
    opHouseholder(h)

Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h h') x`.
"""
function opHouseholder(h :: AbstractVector{T}) where T
  n = length(h)
  prod = @closure v -> (v - 2 * dot(h, v) * h)  # tprod will be inferred
  LinearOperator{T}(n, n, isreal(h), true, prod, nothing, prod)
end

"""
    opHermitian(d, A)

A symmetric/hermitian operator based on the diagonal `d` and lower triangle of `A`.
"""
function opHermitian(d :: AbstractVector{S}, A :: AbstractMatrix{T}) where {S, T}
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  prod = @closure v -> (d .* v + L * v + (v' * L)')[:]
  LinearOperator{U}(m, m, isreal(A), true, prod, nothing, nothing)
end


"""
    opHermitian(A)

A symmetric/hermitian operator based on a matrix.
"""
function opHermitian(T :: AbstractMatrix)
  d = diag(T)
  opHermitian(d, T)
end
