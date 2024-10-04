export check_ctranspose, check_hermitian, check_positive_definite, normest, solve_shifted_system!, ldiv!
import LinearAlgebra.ldiv!

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
  solve_shifted_system!(x, B,  b, σ)

Solve linear system (B + σI) x = b, where B is a forward L-BFGS operator and σ ≥ 0.

### Parameters

- `x::AbstractVector{T}`: preallocated vector of length n that is used to store the solution x.
- `B::LBFGSOperator`: forward L-BFGS operatorthat models a matrix of size n x n.
- `b::AbstractVector{T}`: right-hand side vector of length n.
- `σ::T`: nonnegative shift.

### Returns

- `x::AbstractVector{T}`: solution vector `x` of length n.

### Method

The method uses a two-loop recursion-like approach with modifications to handle the shift `σ`.

### References

Erway, J. B., Jain, V., & Marcia, R. F. Shifted L-BFGS Systems. Optimization Methods and Software, 29(5), pp. 992-1004, 2014.
"""
function solve_shifted_system!(
  x::AbstractVector{T},
  B::LBFGSOperator{T, I, F1, F2, F3},
  b::AbstractVector{T},
  σ::T,
  ) where {T, I, F1, F2, F3}

  if σ < 0
    throw(ArgumentError("σ must be nonnegative"))
  end
  data = B.data
  insert = data.insert
  
  γ_inv = 1 / data.scaling_factor
  x_0 = 1 / (γ_inv + σ)
  @. x = x_0 * b

  max_i = 2 * data.mem
  sign_i = 1

  for i = 1:max_i
    j = (i + 1) ÷ 2
    k = mod(insert + j - 1, data.mem) + 1
    data.shifted_u .= ((sign_i == -1) ? data.b[k] : data.a[k])

    @. data.shifted_p[:, i] = x_0 * data.shifted_u

    sign_t = 1
    for t = 1:(i - 1)
      c0 = dot(view(data.shifted_p, :, t), data.shifted_u)
      c1= sign_t .*data.shifted_v[t]
      c2 = c1 * c0
      view(data.shifted_p, :, i) .+= c2 .* view(data.shifted_p, :, t)
      sign_t = -sign_t
    end

    data.shifted_v[i] = 1 / (1 - sign_i * dot(data.shifted_u, view(data.shifted_p, :, i)))
    x .+= sign_i  *data.shifted_v[i] * (view(data.shifted_p, :, i)' * b) .* view(data.shifted_p, :, i)
    sign_i = -sign_i
  end
  return x
end


"""
    ldiv!(x, B, b)

Solves the linear system defined by the L-BFGS operator `B` and the right-hand side vector `b`, storing the solution in the vector `x`.

### Arguments:

- `x::AbstractVector{T}`: The solution vector, which will be modified in-place.
- `B::LBFGSOperator{T, I, F1, F2, F3}`: The L-BFGS operator that defines the linear system.
- `b::AbstractVector{T}`: The right-hand side vector of the linear system.

### Returns:

- `x::AbstractVector{T}`: The modified solution vector containing the solution to the linear system.

### Examples:

```julia

# Create an L-BFGS operator
B = LBFGSOperator(10)

# Generate random vectors
x = rand(10)
b = rand(10)

# Solve the linear system
ldiv!(x, B, b)

# The vector `x` now contains the solution
"""

function ldiv!(x::AbstractVector{T}, B::LBFGSOperator{T, I, F1, F2, F3}, b::AbstractVector{T}) where {T, I, F1, F2, F3}
  # Call solve_shifted_system! with σ = 0
  solve_shifted_system!(x, B, b, T(0.0))
  return x
end