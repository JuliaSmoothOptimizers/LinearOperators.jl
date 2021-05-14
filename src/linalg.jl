export opInverse, opCholesky, opLDL, opHouseholder, opHermitian 

function mulFact(res, F, v, α, β)
  res .= α .* (F \ v) .+ β .* res 
end

function tmulFact(res, F, u, α, β)
  res .= α .* conj.(F \ conj.(u)) .+ β .* res 
end

"""
    opInverse(Mv, Mtu, Maw, M; symm=false, herm=false)
    opInverse(M; symm=false, herm=false)

Inverse of a matrix as a linear operator using `\\` with storage vectors `Mv`, `Mtu`, `Maw`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`.
This Operator is not in-place when using `mul!`.
"""
function opInverse(Mv::AbstractVector{T}, Mtu::AbstractVector{T}, Maw::AbstractVector{T}, M::AbstractMatrix{T}; 
                   symm = false, herm = false) where {T}
  prod = @closure (res, v, α, β) -> mulFact(res, M, v, α, β)
  tprod = @closure (res, u, α, β) -> mulFact(res, transpose(M), u, α, β)
  ctprod = @closure (res, w, α, β) -> mulFact(res, adjoint(M), w, α, β)
  LinearOperator{T}(size(M, 2), size(M, 1), symm, herm, prod, tprod, ctprod, Mv, Mtu, Maw)
end

function opInverse(M::AbstractMatrix{T}; symm = false, herm = false) where {T}
  Mv = zeros(T, size(M, 2))
  Mtu = symm ? Mv : zeros(T, size(M, 1))
  Maw = herm ? Mv : zeros(T, size(M, 1))
  return opInverse(Mv, Mtu, Maw, M, symm = symm, herm = herm) 
end

"""
    opCholesky(Mv, M, [check=false])
    opCholesky(M, [check=false])

Inverse of a Hermitian and positive definite matrix as a linear operator
using its Cholesky factorization with storage vector `Mv`. 
The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks.
This Operator is not in-place when using `mul!`.
"""
function opCholesky(Mv::AbstractVector{T}, M::AbstractMatrix; check::Bool = false) where {T}
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
    check_positive_definite(M) || throw(LinearOperatorException("matrix is not positive definite"))
  end
  LL = cholesky(M)
  prod = @closure (res, v, α, β) -> mulFact(res, LL, v, α, β)
  tprod = @closure (res, u, α, β) -> tmulFact(res, LL, u, α, β)  # M.' = conj(M)
  ctprod = @closure (res, w, α, β) -> mulFact(res, LL, w, α, β)
  S = eltype(LL)
  LinearOperator{S}(m, m, isreal(M), true, prod, tprod, ctprod, Mv, Mv, Mv)
  #TODO: use iterative refinement.
end

opCholesky(M::AbstractMatrix; check::Bool = false) = opCholesky(zeros(size(M,1)), M; check = check)

"""
    opLDL(Mv, M, [check=false])
    opLDL(M, [check=false])

Inverse of a symmetric matrix as a linear operator using its LDL' factorization
if it exists with storage vector `Mv`. The factorization is computed only once. The optional `check`
argument will perform a cheap hermicity check.
"""
function opLDL(Mv::AbstractVector{T}, M::AbstractMatrix; check::Bool = false) where {T}
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldlt(M)
  prod = @closure (res, v, α, β) -> mulFact(res, LDL, v, α, β)
  tprod = @closure (res, u, α, β) -> tmulFact(res, LDL, u, α, β)  # M.' = conj(M)
  ctprod = @closure (res, w, α, β) -> mulFact(res, LDL, w, α, β)
  S = eltype(LDL)
  return LinearOperator{S}(m, m, isreal(M), true, prod, tprod, ctprod, Mv, Mv, Mv)
  #TODO: use iterative refinement.
end

opLDL(M::AbstractMatrix; check::Bool = false) = opLDL(zeros(size(M,1)), M; check = check)

function mulHouseholder!(res, h, v, α, β)
  res .= α .* (v .- 2 * dot(h, v) .* h) .+ β .* res
end

"""
    opHouseholder(Mv, Mtu, h)
    opHouseholder(h)

Apply a Householder transformation defined by the vector `h` with storage vectors Mv, Mtu.
The result is `x -> (I - 2 h h') x`.
"""
function opHouseholder(Mv::AbstractVector{T}, Mtu::AbstractVector{T}, h::AbstractVector{T}) where {T}
  n = length(h)
  prod! = @closure (res, v, α, β) -> mulHouseholder!(res, h, v, α, β)  # tprod will be inferred
  LinearOperator{T}(n, n, isreal(h), true, prod!, nothing, prod!, Mv, Mtu, Mv)
end

function opHouseholder(h::AbstractVector{T}) where {T} 
  Mv = similar(h, length(h))
  Mtu = isreal(h) ? Mv : copy(Mv)
  return opHouseholder(Mv, Mtu, h)
end

function mulHermitian(res, d, L, v,  α, β)
  res .= α .* (d .* v .+ L * v .+ (v' * L)')[:] .+ β .* res
end

"""
    opHermitian(Mv, d, A)
    opHermitian(d, A)

A symmetric/hermitian operator based on the diagonal `d` and lower triangle of `A`.
"""
function opHermitian(Mv::AbstractVector{T}, d::AbstractVector{S}, A::AbstractMatrix{T}) where {S, T}
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  prod = @closure (res, v, α, β) -> mulHermitian(res, d, L, v, α, β)
  LinearOperator{U}(m, m, isreal(A), true, prod, nothing, nothing, Mv, Mv, Mv)
end

opHermitian(d::AbstractVector{S}, A::AbstractMatrix{T}) where {S, T} = opHermitian(zeros(T, size(A, 1)), d, A)

"""
    opHermitian(A)

A symmetric/hermitian operator based on a matrix.
"""
function opHermitian(T::AbstractMatrix)
  d = diag(T)
  opHermitian(d, T)
end
