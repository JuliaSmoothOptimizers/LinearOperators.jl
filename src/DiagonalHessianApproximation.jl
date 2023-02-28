export DiagonalQN, SpectralGradient

"""
Implementation of the diagonal quasi-Newton approximations described in

M. Zhu, J. L. Nazareth and H. Wolkowicz
The Quasi-Cauchy Relation and Diagonal Updating.
SIAM Journal on Optimization, vol. 9, number 4, pp. 1192-1204, 1999.
https://doi.org/10.1137/S1052623498331793.

and

Andrei, N. 
A diagonal quasi-Newton updating method for unconstrained optimization. 
https://doi.org/10.1007/s11075-018-0562-7
"""
mutable struct DiagonalQN{T <: Real, I <: Integer, V <: AbstractVector{T}, F} <:
               AbstractDiagonalQuasiNewtonOperator{T}
  d::V # Diagonal of the operator
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::F
  ctprod!::F
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
  psb::Bool
end

"""
    DiagonalQN(d)

Construct a linear operator that represents a diagonal quasi-Newton approximation.
The approximation satisfies the weak secant equation and is not guaranteed to be
positive definite.

# Arguments

- `d::AbstractVector`: initial diagonal approximation;
- `psb::Bool`: whether to use the diagonal PSB update or the Andrei update.
"""
function DiagonalQN(d::AbstractVector{T}, psb::Bool = false) where {T <: Real}
  prod = (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β)
  n = length(d)
  DiagonalQN(d, n, n, true, true, prod, prod, prod, 0, 0, 0, true, true, true, psb)
end

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::DiagonalQN{T, I, V, F},
  s0::V,
  y0::V,
) where {T <: Real, I <: Integer, V <: AbstractVector{T}, F}
  s0Norm = norm(s0, 2)
  if s0Norm == 0
    error("Cannot update DiagonalQN operator with s=0")
  end
  # sᵀBs = sᵀy can be scaled by ||s||² without changing the update
  s = (si / s0Norm for si ∈ s0)
  s2 = (si^2 for si ∈ s)
  y = (yi / s0Norm for yi ∈ y0) 
  trA2 = dot(s2, s2)
  sT_y = dot(s, y)
  sT_B_s = dot(s2, B.d)
  q = sT_y - sT_B_s
  if B.psb
    q /= trA2
    B.d .+= q .* s .^ 2
  else
    sT_s = dot(s, s)
    q += sT_s
    q /= trA2
    B.d .+= q .* s .^ 2 .- 1
  end
  return B
end

"""
    reset!(op::DiagonalQN)
Resets the DiagonalQN data of the given operator.
"""
function reset!(op::DiagonalQN{T}) where {T <: Real}
  op.d .= one(T)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end

"""
Implementation of a spectral gradient quasi-Newton approximation described in

Birgin, E. G., Martínez, J. M., & Raydan, M. 
Spectral Projected Gradient Methods: Review and Perspectives. 
https://doi.org/10.18637/jss.v060.i03
"""
mutable struct SpectralGradient{T <: Real, I <: Integer, F} <:
               AbstractDiagonalQuasiNewtonOperator{T}
  d::Vector{T} # Diagonal coefficient of the operator (multiple of the identity)
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::F
  ctprod!::F
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

"""
        SpectralGradient(σ, n)

Construct a spectral gradient Hessian approximation.
The approximation is defined as σI.

# Arguments

- `σ::Real`: initial positive multiple of the identity;
- `n::Int`: operator size.
"""
function SpectralGradient(σ::T, n::I) where {T <: Real, I <: Integer}
  @assert σ > 0
  d = [σ]
  prod = (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β)
  SpectralGradient(d, n, n, true, true, prod, prod, prod, 0, 0, 0, true, true, true)
end

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::SpectralGradient{T, I, F},
  s::V,
  y::V,
) where {T <: Real, I <: Integer, F, V <: AbstractVector{T}}
  if all(s .== 0)
    error("Cannot divide by zero and s .= 0")
  end
  B.d[1] = dot(s, y) / dot(s, s)
  return B
end

"""
    reset!(op::SpectralGradient)
Resets the SpectralGradient data of the given operator.
"""
function reset!(op::SpectralGradient{T}) where {T <: Real}
  op.d[1] = one(T)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
