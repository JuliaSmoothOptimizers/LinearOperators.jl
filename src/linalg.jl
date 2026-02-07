export opInverse, opCholesky, opHouseholder, opLDL, opHermitian

function mulFact!(res, F, v, α, β::T) where {T}
  if β == zero(T)
    res .= α .* (F \ v)
  else
    res .= α .* (F \ v) .+ β .* res
  end
end

function tmulFact!(res, F, u, α, β::T) where {T}
  if β == zero(T)
    res .= α .* conj!(F \ conj.(u))
  else
    res .= α .* conj!(F \ conj.(u)) .+ β .* res
  end
end

"""
    opInverse(M; symm=false, herm=false)

Inverse of a matrix as a linear operator using `\\`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`.
This Operator is not in-place when using `mul!`.
"""
function opInverse(M::AbstractMatrix{T}; symm = false, herm = false) where {T}
  prod! = @closure (res, v, α, β) -> mulFact!(res, M, v, α, β)
  tprod! = @closure (res, u, α, β) -> mulFact!(res, transpose(M), u, α, β)
  ctprod! = @closure (res, w, α, β) -> mulFact!(res, adjoint(M), w, α, β)
  LinearOperator{T, Vector{T}}(size(M, 2), size(M, 1), symm, herm, prod!, tprod!, ctprod!)
end

"""
    opCholesky(M, [check=false])

Inverse of a Hermitian and positive definite matrix as a linear operator
using its Cholesky factorization.
The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks.
This Operator is not in-place when using `mul!`.
"""
function opCholesky(M::AbstractMatrix; check::Bool = false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
    check_positive_definite(M) || throw(LinearOperatorException("matrix is not positive definite"))
  end
  LL = cholesky(M)
  prod! = @closure (res, v, α, β) -> mulFact!(res, LL, v, α, β)
  tprod! = @closure (res, u, α, β) -> tmulFact!(res, LL, u, α, β)  # M.' = conj(M)
  ctprod! = @closure (res, w, α, β) -> mulFact!(res, LL, w, α, β)
  S = eltype(LL)
  LinearOperator{S, Vector{S}}(m, m, isreal(M), true, prod!, tprod!, ctprod!)
  #TODO: use iterative refinement.
end

"""
    opLDL(M, [check=false])

Inverse of a symmetric matrix as a linear operator using its LDLᵀ factorization
if it exists. The factorization is computed only once. The optional `check`
argument will perform a cheap hermicity check.

If M is sparse and real, then only the upper triangle should be stored in order to use
[`LDLFactorizations.jl`](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl):

    using LDLFactorizations
    triu!(M)
    opLDL(Symmetric(M, :U))

"""
function opLDL end

function mulHouseholder!(res, h, v, α, β::T) where {T}
  if β == zero(T)
    res .= α .* (v .- 2 * dot(h, v) .* h)
  else
    res .= α .* (v .- 2 * dot(h, v) .* h) .+ β .* res
  end
end

"""
    opHouseholder(h)

Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h hᵀ) x`.
"""
function opHouseholder(h::AbstractVector{T}) where {T}
  n = length(h)
  prod! = @closure (res, v, α, β) -> mulHouseholder!(res, h, v, α, β)  # tprod will be inferred
  LinearOperator{T, Vector{T}}(n, n, isreal(h), true, prod!, nothing, prod!)
end

function mulHermitian!(res, d, L, v, α, β::T) where {T}
  if β == zero(T)
    res .= α .* (d .* v .+ L * v .+ (v' * L)')[:]
  else
    res .= α .* (d .* v .+ L * v .+ (v' * L)')[:] .+ β .* res
  end
end

"""
    opHermitian(d, A)

A symmetric/hermitian operator based on the diagonal `d` and lower triangle of `A`.
"""
function opHermitian(d::AbstractVector{S}, A::AbstractMatrix{T}) where {S, T}
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  prod! = @closure (res, v, α, β) -> mulHermitian!(res, d, L, v, α, β)
  LinearOperator{U, Vector{U}}(m, m, isreal(A), true, prod!, nothing, nothing)
end

"""
    opHermitian(A)

A symmetric/hermitian operator based on a matrix.
"""
function opHermitian(T::AbstractMatrix)
  d = diag(T)
  opHermitian(d, T)
end
