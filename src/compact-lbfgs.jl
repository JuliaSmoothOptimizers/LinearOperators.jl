export CompactLBFGSOperator

mutable struct CompactLBFGSData{T,V<:AbstractVector{T},MT<:AbstractMatrix{T}}
  const scaling::Bool
  ξ::T
  Sk::MT # n x p
  Yk::MT # n x p
  Lk::MT # p x p
  _Dkinvsq::V  # p
  _DLk::MT # p x p
  Mk::MT # p x p
  Uk::MT # n x p
  Vk::MT # n x p
  _y::V  # p
  _mem::Int
  _insert::Int
  _nskip::Int
  _max_skip::Int
  const damped::Bool
  σ₂::T
  σ₃::T
  _Bs::V # n
end

function CompactLBFGSData(
  T::Type,
  n::I;
  mem::I = 6,
  scaling::Bool = true,
  damped::Bool = false,
  σ₂::Float64 = 0.99,
  σ₃::Float64 = 10.0,
  max_skip = 2,
) where {I<:Integer}
  mem = max(mem, 1)
  return CompactLBFGSData{T,Vector{T},Matrix{T}}(
    scaling,
    one(T),
    zeros(T, n, mem),
    zeros(T, n, mem),
    zeros(T, mem, mem),
    zeros(T, mem),
    zeros(T, mem, mem),
    zeros(T, mem, mem),
    zeros(T, n, mem),
    zeros(T, n, mem),
    zeros(T, mem),
    mem,
    1,
    0,
    max_skip,
    damped,
    convert(T, σ₂),
    convert(T, σ₃),
    zeros(T, n),
  )
end

function LinearAlgebra.mul!(
  x::AbstractVector,
  data::CompactLBFGSData,
  y::AbstractVector,
  α::Real,
  β::Real,
)
  if iszero(β)
    x .= (α*data.ξ) .* y                                    # x = α * (ξI)y
  else
    x .= β .* x .+ (α*data.ξ) .* y                          # x = βx + α * (ξI)y
  end

  k = min(data._insert - 1, data._mem)

  @views mul!(data._y[1:k], data.Uk[:, 1:k]', y)            # _y = Uₖᵀy
  @views mul!(x, data.Uk[:, 1:k], data._y[1:k], -α, one(α)) #  x = βx + α * (ξI - UₖUₖᵀ)y
  @views mul!(data._y[1:k], data.Vk[:, 1:k]', y)            # _y = Vₖᵀy
  @views mul!(x, data.Vk[:, 1:k], data._y[1:k], α, one(α))  #  x = βx + α * (ξI - UₖUₖᵀ + VₖVₖᵀ)y
end

function reset!(data::CompactLBFGSData)
  data.ξ = 1
  data.Sk .= 0
  data.Yk .= 0
  data.Lk .= 0
  data.Mk .= 0
  data.Uk .= 0
  data.Vk .= 0
  data._Dkinvsq .= 0
  data._DLk .= 0
  data._y .= 0

  data._insert = 1
  data._nskip = 0

end
# Powell's damping: return y, or θy + (1 - θ)Bs stored in data._Bs when sᵀy is
# too small or too large compared with sᵀBs.
function damp!(data::CompactLBFGSData{T,V,MT}, s::V, y::V) where {T,V,MT}
  Bs = data._Bs
  mul!(Bs, data, s, one(T), zero(T))
  sBs = dot(s, Bs)
  sy = dot(s, y)
  if sy < (1 - data.σ₂) * sBs
    θ = data.σ₂ * sBs / (sBs - sy)
  elseif sy > (1 + data.σ₃) * sBs
    θ = data.σ₃ * sBs / (sy - sBs)
  else
    return y
  end
  Bs .= θ .* y .+ (1 - θ) .* Bs
  return Bs
end

function Base.push!(data::CompactLBFGSData{T,V,MT}, s::V, y::V) where {T,V,MT}
  if data.damped
    y = damp!(data, s, y)
  end

  k, mem = data._insert, data._mem
  Sk, Yk, Lk, Dkinvsq, _DLk, Mk, Uk, Vk =
    data.Sk, data.Yk, data.Lk, data._Dkinvsq, data._DLk, data.Mk, data.Uk, data.Vk
  ξ = data.ξ

  sy = dot(s, y)

  if sy <= eps(T)
    data._nskip = data._nskip + 1
    if data._nskip > data._max_skip
      reset!(data)
    end
    return
  else
    data._nskip = 0
  end

  if data.scaling
    ξ = data.ξ = dot(y, y) / sy
  end

  # Shift
  if k > mem
    @inbounds for i = 1:(mem-1)
      copyto!(view(Sk, :, i), view(Sk, :, i+1))
      copyto!(view(Yk, :, i), view(Yk, :, i+1))
      Dkinvsq[i] = Dkinvsq[i+1]

      for j = 1:(i-1)
        Lk[i, j] = Lk[i+1, j+1]
      end
    end
    k = mem
  end

  copyto!(view(Sk, :, k), s)
  copyto!(view(Yk, :, k), y)

  @inbounds for j = 1:(k-1)
    Lk[k, j] = dot(s, view(Yk, :, j))
  end
  Dkinvsq[k] = 1 / sqrt(sy)

  # Compute compact representation Bₖ = σₖI - UₖUₖᵀ + VₖVₖᵀ.
  # Source: https://github.com/MadNLP/MadNLP.jl/blob/v0.9.1/src/quasi_newton.jl#L366

  # Step 1: compute Mₖ
  mul!(_DLk, Diagonal(Dkinvsq), Lk')
  mul!(Mk, _DLk', _DLk)                                    # Mₖ = Lₖ Dₖ⁻¹ Lₖᵀ
  mul!(Mk, Sk', Sk, ξ, one(T))                             # Mₖ = ξ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ

  # Step 2: factorize Mₖ
  try
    cholesky!(Symmetric(@view Mk[1:k, 1:k]))                 # Mₖ = Jₖᵀ Jₖ (factorization)
  catch e
    @warn "Cholesky factorization of Mₖ failed. Resetting the compact BFGS model."
    reset!(data)
    return
  end

  # Step 3: compute Vₖ
  mul!(Vk, Yk, Diagonal(Dkinvsq))

  # Step 4: compute Uₖ
  mul!(Uk, Vk, _DLk)                                       # Uₖ = Yₖ Dₖ⁻¹ Lₖᵀ
  axpy!(ξ, Sk, Uk)                                         # Uₖ = ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ
  @views rdiv!(Uk[:, 1:k], UpperTriangular(Mk[1:k, 1:k]))  # Uₖ = (ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ) Jₖ⁻¹

  data._insert = min(k + 1, mem + 1)
end

"A type for limited-memory BFGS approximations in compact form."
mutable struct CompactLBFGSOperator{T, I <: Integer, F, Ft, Fct, D <: CompactLBFGSData{T}} <:
               AbstractQuasiNewtonOperator{T}
  const nrow::I
  const ncol::I
  const symmetric::Bool
  const hermitian::Bool
  const prod!::F    # apply the operator to a vector
  const tprod!::Ft    # apply the transpose operator to a vector
  const ctprod!::Fct   # apply the transpose conjugate operator to a vector
  const data::D
  nprod::I
  ntprod::I
  nctprod::I
end

CompactLBFGSOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  data::D,
) where {T, I <: Integer, F, Ft, Fct, D} = CompactLBFGSOperator{T, I, F, Ft, Fct, D}(
  nrow,
  ncol,
  symmetric,
  hermitian,
  prod!,
  tprod!,
  ctprod!,
  data,
  0,
  0,
  0,
)

has_args5(op::CompactLBFGSOperator) = true
isallocated5(op::CompactLBFGSOperator) = true
storage_type(op::CompactLBFGSOperator) = typeof(op.data._y)

"""
    CompactLBFGSOperator(T, n; [mem = 6, scaling = true, damped = false, max_skip = 2])
    CompactLBFGSOperator(n; kwargs...)

Construct a limited-memory BFGS approximation stored in compact form
``B = ξI - UUᵀ + VVᵀ``. If the type `T` is omitted, then `Float64` is used.

# Keyword arguments
- `mem::Int = 6`: number of pairs ``(s, y)`` stored.
- `scaling::Bool = true`: whether the initial approximation is ``B_0 = ξ I`` with ``ξ = y^T y / s^T y``.
- `damped::Bool = false`: whether to apply Powell's damping to the pairs ``(s, y)``, with parameters `σ₂::Float64 = 0.99` and `σ₃::Float64 = 10.0`.
- `max_skip::Int = 2`: an update is skipped when `dot(s, y) ≤ eps(T)`; the approximation is reset once more than `max_skip` consecutive updates have been skipped.
"""
function CompactLBFGSOperator(T::Type, n::I; kwargs...) where {I <: Integer}
  data = CompactLBFGSData(T, n; kwargs...)
  prod! = @closure (res, x, α, β) -> mul!(res, data, x, α, β)
  return CompactLBFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, data)
end

CompactLBFGSOperator(n::Integer; kwargs...) = CompactLBFGSOperator(Float64, n; kwargs...)

function push!(op::CompactLBFGSOperator, s::AbstractVector, y::AbstractVector)
  push!(op.data, s, y)
  return op
end

function reset!(op::CompactLBFGSOperator)
  reset!(op.data)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end

# Copying API
function Base.copy(data::CompactLBFGSData{T,V,MT}) where {T,V,MT}
  return CompactLBFGSData(
    data.scaling,
    data.ξ,
    copy(data.Sk),
    copy(data.Yk),
    copy(data.Lk),
    copy(data._Dkinvsq),
    copy(data._DLk),
    copy(data.Mk),
    copy(data.Uk),
    copy(data.Vk),
    copy(data._y),
    data._mem,
    data._insert,
    data._nskip,
    data._max_skip,
    data.damped,
    data.σ₂,
    data.σ₃,
    copy(data._Bs),
  )
end

function Base.similar(data::CompactLBFGSData{T,V,MT}) where {T,V,MT}
  return CompactLBFGSData(
    data.scaling,
    data.ξ,
    similar(data.Sk),
    similar(data.Yk),
    similar(data.Lk),
    similar(data._Dkinvsq),
    similar(data._DLk),
    similar(data.Mk),
    similar(data.Uk),
    similar(data.Vk),
    similar(data._y),
    data._mem,
    data._insert,
    data._nskip,
    data._max_skip,
    data.damped,
    data.σ₂,
    data.σ₃,
    similar(data._Bs),
  )
end

function Base.copy!(dest::CompactLBFGSData{T,V,MT}, src::CompactLBFGSData{T,V,MT}) where {T,V,MT}
  @assert dest.scaling == src.scaling
  @assert dest.damped == src.damped
  dest.ξ = src.ξ
  copyto!(dest.Sk, src.Sk)
  copyto!(dest.Yk, src.Yk)
  copyto!(dest.Lk, src.Lk)
  copyto!(dest._Dkinvsq, src._Dkinvsq)
  copyto!(dest._DLk, src._DLk)
  copyto!(dest.Mk, src.Mk)
  copyto!(dest.Uk, src.Uk)
  copyto!(dest.Vk, src.Vk)
  copyto!(dest._y, src._y)
  dest._mem = src._mem
  dest._insert = src._insert
  dest._nskip = src._nskip
  dest._max_skip = src._max_skip
  dest.σ₂ = src.σ₂
  dest.σ₃ = src.σ₃
end