export DiagonalQN, SpectralGradient

"""
Implementation of the diagonal quasi-Newton approximation described in

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
end

"""
    DiagonalQN(d)

Construct a linear operator that represents a diagonal quasi-Newton approximation.
The approximation satisfies the weak secant equation and is not guaranteed to be
positive definite.

# Arguments

- `d::AbstractVector`: initial diagonal approximation.
"""
function DiagonalQN(d::AbstractVector{T}) where {T <: Real}
  prod = (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β)
  DiagonalQN(d, length(d), length(d), true, true, prod, prod, prod, 0, 0, 0, true, true, true)
end

# update function
# s = x_{k+1} - x_k
# y = ∇f(x_{k+1}) - ∇f(x_k)
function push!(
  B::DiagonalQN{T, I, V, F},
  s::V,
  y::V,
) where {T <: Real, I <: Integer, V <: AbstractVector{T}, F}
  trA2 = zero(T)
  for i in eachindex(s)
    trA2 += s[i]^4
  end
  sT_s = dot(s, s)
  sT_y = dot(s, y)
  sT_B_s = sum(s[i]^2 * B.d[i] for i ∈ eachindex(s))
  if trA2 == 0
    error("Cannot divide by zero and trA2 = 0")
  end
  q = (sT_y + sT_s - sT_B_s) / trA2
  B.d .+= q .* s .^ 2 .- 1
  return B
end

"""
Implementation of a spectral gradient quasi-Newton approximation described in

Birgin, E. G., Martínez, J. M., & Raydan, M. 
Spectral Projected Gradient Methods: Review and Perspectives. 
https://doi.org/10.18637/jss.v060.i03
"""
mutable struct SpectralGradient{T <: Real, I <: Integer, F} <:
               AbstractDiagonalQuasiNewtonOperator{T}
  d::T # Diagonal coefficient of the operator (multiple of the identity)
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
function SpectralGradient(d::T, n::I) where {T <: Real, I <: Integer}
  @assert d > 0
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
  B.d = dot(s, y) / dot(s, s)
  return B
end
