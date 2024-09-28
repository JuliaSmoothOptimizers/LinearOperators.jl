export check_ctranspose, check_hermitian, check_positive_definite, normest, solve_shifted_system!

"""
  normest(S) estimates the matrix 2-norm of S.
  This function is an adaptation of Matlab's built-in NORMEST.
  This method allocates.

  -----------------------------------------
  Inputs:
    S --- Matrix or LinearOperator type, 
    tol ---  relative error tol, default(or -1) Machine eps
    maxiter --- maximum iteration, default 100
    
  Returns:
    e --- the estimated norm
    cnt --- the number of iterations used
  """
function normest(S, tol = -1, maxiter = 100)
  (m, n) = size(S)
  cnt = 0
  if tol == -1
    tol = Float64(eps(eltype(S)))
  end
  # Compute an "estimate" of the ab-val column sums.
  v = ones(eltype(S), m)
  v[randn(m) .< 0] .= -1
  x = zeros(eltype(S), n)
  mul!(x, S', v)
  e = norm(x)

  if e == 0
    return e, cnt
  end

  x ./= e
  e_0 = zero(e)
  Sx = zeros(eltype(S), n)

  while abs(e - e_0) > tol * e
    e_0 = e
    mul!(Sx, S, x)
    if count(x -> x != 0, Sx) == 0
      Sx .= randn(eltype(Sx), size(Sx))
    end
    mul!(x, S', Sx)
    normx = norm(x)
    e = normx / norm(Sx)
    x ./= normx
    cnt = cnt + 1
    if cnt > maxiter
      @warn("normest did not converge ", maxiter, tol,)
      break
    end
  end

  return e, cnt
end

"""
    check_ctranspose(op)

Cheap check that the operator and its conjugate transposed are related.
"""
function check_ctranspose(op::AbstractLinearOperator{T}) where {T <: Union{AbstractFloat, Complex}}
  (m, n) = size(op)
  x = rand(n)
  y = rand(m)
  yAx = dot(y, op * x)
  xAty = dot(x, op' * y)
  ε = eps(real(eltype(op)))
  return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1 / 3)
end

function check_ctranspose(op::AbstractLinearOperator{T}) where {T <: Integer}
  (m, n) = size(op)
  x = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  y = convert(Vector{T}, (floor.(10 * rand(m)))) .- 5
  yAx = dot(y, op * x)
  xAty = dot(x, op' * y)
  return yAx == xAty
end

check_ctranspose(M::AbstractMatrix) = check_ctranspose(LinearOperator(M))

"""
    check_hermitian(op)

Cheap check that the operator is Hermitian.
"""
function check_hermitian(op::AbstractLinearOperator{T}) where {T <: Union{AbstractFloat, Complex}}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = rand(n)
  w = copy(op * v)  # copy necessary to guard against in-place operators
  s = dot(w, w)  # = (Av)'(Av) = v' A' A v.
  y = op * w
  t = dot(v, y)  # = v' A A v.
  ε = eps(real(eltype(op)))
  return abs(s - t) < (abs(s) + ε) * ε^(1 / 3)
end

function check_hermitian(op::AbstractLinearOperator{T}) where {T <: Integer}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  w = copy(op * v)
  s = dot(w, w)  # = (Av)'(Av) = v' A' A v.
  y = op * w
  t = dot(v, y)  # = v' A A v.
  return s == t
end

check_hermitian(M::AbstractMatrix) = check_hermitian(LinearOperator(M))

"""
    check_positive_definite(op; semi=false)

Cheap check that the operator is positive (semi-)definite.
"""
function check_positive_definite(
  op::AbstractLinearOperator{T};
  semi = false,
) where {T <: Union{AbstractFloat, Complex}}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = rand(n)
  w = op * v
  vw = dot(v, w)
  ε = eps(real(eltype(op)))
  if imag(vw) > sqrt(ε) * abs(vw)
    return false
  end
  vw = real(vw)
  return semi ? (vw ≥ 0) : (vw > 0)
end

function check_positive_definite(op::AbstractLinearOperator{T}; semi = false) where {T <: Integer}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  w = op * v
  vw = dot(v, w)
  return semi ? (vw ≥ 0) : (vw > 0)
end

check_positive_definite(M::AbstractMatrix; kwargs...) =
  check_positive_definite(LinearOperator(M); kwargs...)


  """
  solve_shifted_system!(op::LBFGSOperator{T,I,F1,F2,F3}, 
                        z::AbstractVector{T}, 
                        σ::T, 
                        γ_inv::T, 
                        inv_Cz::AbstractVector{T}, 
                        p::AbstractVector{T}, 
                        v::AbstractVector{T}, 
                        u::AbstractVector{T}) where {T,I,F1,F2,F3}

Solves linear system (B + σI) x = b, where B is a forward L-BFGS operator and σ ≥ 0.

### Parameters

- `B::LBFGSOperator{T,I,F1,F2,F3}`: forward L-BFGS operator.
- `z::AbstractVector{T}`: The vector representing `-b`.
- `σ::T`: Nonnegative shift.
- `inv_Cz::AbstractVector{T}`: A preallocated vector used to store the solution.

### Returns

- `inv_Cz::AbstractVector{T}`: The solution vector `s` such that `(B_k + σ * I) s = -∇f(x_k)`.

### Method

The method uses a two-loop recursion-like approach with modifications to handle the shift `σ`.

### References
Erway, J. B., Jain, V., & Marcia, R. F. (2013). Shifted L-BFGS Systems.
"""

function solve_shifted_system!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  z::AbstractVector{T},
  σ::T,
  inv_Cz::AbstractVector{T},
  ) where {T, I, F1, F2, F3}
  data = op.data
  insert = data.insert
  
  γ_inv = 1 / data.scaling_factor
  inv_c0 = 1 / (γ_inv + σ)
  @. inv_Cz = inv_c0 * z

  max_i = 2 * data.mem
  for i = 1:max_i
    j = (i + 1) ÷ 2
    k = mod(insert + j - 1, data.mem) + 1
    data.shifted_u .= ((i % 2) == 0 ? data.b[k] : data.a[k])

    @. data.shifted_p[:, i] = inv_c0 * data.shifted_u

    for t = 1:(i - 1)
      c0 = dot(view(data.shifted_p, :, t), data.shifted_u)
      c1= ((t % 2) == 0 ? -1 : 1) .*data.shifted_v[t]
      c2 = c1 * c0
      view(data.shifted_p, :, i) .+= c2 .* view(data.shifted_p, :, t)
    end

   data.shifted_v[i] = 1 / (1 + ((i % 2) == 0 ? 1 : -1) * dot(data.shifted_u, view(data.shifted_p, :, i)))
    inv_Cz .+= ((i % 2) == 0 ? -1 : 1)  *data.shifted_v[i] * (view(data.shifted_p, :, i)' * z) .* view(data.shifted_p, :, i)
  end
  return inv_Cz
end
