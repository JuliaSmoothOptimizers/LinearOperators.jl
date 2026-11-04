export CompactLBFGSOperator

"A data type to hold information relative to compact LBFGS operators."
mutable struct CompactLBFGSData{T, V <: AbstractVector{T}, MT <: AbstractMatrix{T}}
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
  mem::I = 5,
  scaling::Bool = true,
  damped::Bool = false,
  σ₂::Float64 = 0.99,
  σ₃::Float64 = 10.0,
  max_skip::Integer = typemax(Int),
  S::Type = Vector{T},
) where {I <: Integer}
  mem = max(mem, 1)
  Sk = fill!(similar(S(undef, 0), T, n, mem), zero(T))
  return CompactLBFGSData{T, S, typeof(Sk)}(
    scaling,
    one(T),
    Sk,
    fill!(similar(S(undef, 0), T, n, mem), zero(T)),
    fill!(similar(S(undef, 0), T, mem, mem), zero(T)),
    fill!(S(undef, mem), zero(T)),
    fill!(similar(S(undef, 0), T, mem, mem), zero(T)),
    fill!(similar(S(undef, 0), T, mem, mem), zero(T)),
    fill!(similar(S(undef, 0), T, n, mem), zero(T)),
    fill!(similar(S(undef, 0), T, n, mem), zero(T)),
    fill!(S(undef, mem), zero(T)),
    mem,
    1,
    0,
    max_skip,
    damped,
    convert(T, σ₂),
    convert(T, σ₃),
    fill!(S(undef, n), zero(T)),
  )
end

CompactLBFGSData(n::I; kwargs...) where {I <: Integer} = CompactLBFGSData(Float64, n; kwargs...)

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

# Allow for GPU-compatible implementation:
# GPU arrays can override these methods to avoid views throughout the implementation.
_active(v::AbstractVector, k) = view(v, 1:k)
_active(A::AbstractMatrix, k) = view(A, :, 1:k)
_active(A::AbstractMatrix, k, l) = view(A, 1:k, 1:l)

function compact_lbfgs_multiply!(
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
  Uk, Vk, _y = _active(data.Uk, k), _active(data.Vk, k), _active(data._y, k)

  mul!(_y, Uk', y)                                          # _y = Uₖᵀy
  mul!(x, Uk, _y, -α, one(α))                               #  x = βx + α * (ξI - UₖUₖᵀ)y
  mul!(_y, Vk', y)                                          # _y = Vₖᵀy
  mul!(x, Vk, _y, α, one(α))                                #  x = βx + α * (ξI - UₖUₖᵀ + VₖVₖᵀ)y
  return x
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

# Build an operator around existing data.
function CompactLBFGSOperator(data::CompactLBFGSData{T}) where {T}
  n = size(data.Sk, 1)
  prod! = @closure (res, x, α, β) -> compact_lbfgs_multiply!(res, data, x, α, β)
  return CompactLBFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, data)
end

"""
    CompactLBFGSOperator(T, n; [mem=5, scaling=true])
    CompactLBFGSOperator(n; [mem=5, scaling=true])
Construct a limited-memory BFGS approximation in compact form ``B = ξI - UUᵀ + VVᵀ``.
If the type `T` is omitted, then `Float64` is used.

Only the forward form is implemented: there is no inverse compact L-BFGS operator.
Use [`InverseLBFGSOperator`](@ref) for the inverse form.

# Keyword arguments
- `mem::Int = 5`: number of pairs ``(s, y)`` stored.
- `scaling::Bool = true`: whether the initial approximation is ``B_0 = ξ I`` with ``ξ = y^T y / s^T y``.
- `damped::Bool = false`: whether to apply Powell's damping to the pairs ``(s, y)``.
- `σ₂::Float64 = 0.99` and `σ₃::Float64 = 10.0`: damping parameters.
- `max_skip::Int = typemax(Int)`: an update is skipped when `dot(s, y) ≤ eps(T)`;
  the approximation is reset once more than `max_skip` consecutive updates have been skipped;
  deactivated by default.
- `S::Type = Vector{T}`: storage type of the vectors; the matrices have the corresponding matrix type.
"""
CompactLBFGSOperator(T::Type, n::Integer; kwargs...) =
  CompactLBFGSOperator(CompactLBFGSData(T, n; kwargs...))

CompactLBFGSOperator(n::Integer; kwargs...) = CompactLBFGSOperator(Float64, n; kwargs...)

# Powell's damping: return y, or θy + (1 - θ)Bs stored in op.data._Bs when sᵀy is
# too small or too large compared with sᵀBs.
function damp!(op::CompactLBFGSOperator{T}, s::AbstractVector{T}, y::AbstractVector{T}) where {T}
  data = op.data
  Bs = data._Bs
  mul!(Bs, op, s, one(T), zero(T))
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

"""
    push!(op, s, y)

Push a new {s,y} pair into a compact L-BFGS operator.
"""
function push!(op::CompactLBFGSOperator{T}, s::AbstractVector{T}, y::AbstractVector{T}) where {T}
  data = op.data
  if data.damped
    y = damp!(op, s, y)
  end

  k, mem = data._insert, data._mem
  Sk, Yk, Lk, Dkinvsq, _DLk, Mk, Uk, Vk =
    data.Sk, data.Yk, data.Lk, data._Dkinvsq, data._DLk, data.Mk, data.Uk, data.Vk
  ξ = data.ξ

  sy = dot(s, y)

  if !(sy > eps(T))
    data._nskip = data._nskip + 1
    if data._nskip > data._max_skip
      reset!(data)
    end
    return op
  else
    data._nskip = 0
  end

  if data.scaling
    ξ = data.ξ = dot(y, y) / sy
  end

  # Shift
  if k > mem
    @inbounds for i = 1:(mem - 1)
      copyto!(view(Sk, :, i), view(Sk, :, i+1))
      copyto!(view(Yk, :, i), view(Yk, :, i+1))
      Dkinvsq[i] = Dkinvsq[i + 1]

      for j = 1:(i - 1)
        Lk[i, j] = Lk[i + 1, j + 1]
      end
    end
    k = mem
  end

  copyto!(view(Sk, :, k), s)
  copyto!(view(Yk, :, k), y)

  @inbounds for j = 1:(k - 1)
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
  M = _active(Mk, k, k)
  view(M, diagind(M)) .+= iszero.(_active(Dkinvsq, k)) # Avoid singular M for padded matrices.
  
  F = cholesky!(Symmetric(M), check = false)               # Mₖ = Jₖᵀ Jₖ (factorization)
  
  if !issuccess(F)
    @warn "Cholesky factorization of Mₖ failed. Resetting the compact BFGS model."
    reset!(data)
    return op
  end

  # Step 3: compute Vₖ
  mul!(Vk, Yk, Diagonal(Dkinvsq))

  # Step 4: compute Uₖ
  mul!(Uk, Vk, _DLk)                                       # Uₖ = Yₖ Dₖ⁻¹ Lₖᵀ
  axpy!(ξ, Sk, Uk)                                         # Uₖ = ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ
  rdiv!(_active(Uk, k), UpperTriangular(M))                # Uₖ = (ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ) Jₖ⁻¹

  data._insert = min(k + 1, mem + 1)
  return op
end

"""
    reset!(op)

Reset the compact L-BFGS data and the product counters of the given operator.
"""
function reset!(op::CompactLBFGSOperator)
  reset!(op.data)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end

# Copying API
function Base.copy(op::CompactLBFGSOperator)
  data = op.data
  return CompactLBFGSOperator(
    CompactLBFGSData(
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
    ),
  )
end

function Base.similar(op::CompactLBFGSOperator)
  data = op.data
  return CompactLBFGSOperator(
    CompactLBFGSData(
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
    ),
  )
end

function Base.copy!(dest::CompactLBFGSOperator, src::CompactLBFGSOperator)
  d, s = dest.data, src.data
  @assert d.scaling == s.scaling
  @assert d.damped == s.damped
  d.ξ = s.ξ
  copyto!(d.Sk, s.Sk)
  copyto!(d.Yk, s.Yk)
  copyto!(d.Lk, s.Lk)
  copyto!(d._Dkinvsq, s._Dkinvsq)
  copyto!(d._DLk, s._DLk)
  copyto!(d.Mk, s.Mk)
  copyto!(d.Uk, s.Uk)
  copyto!(d.Vk, s.Vk)
  copyto!(d._y, s._y)
  d._mem = s._mem
  d._insert = s._insert
  d._nskip = s._nskip
  d._max_skip = s._max_skip
  d.σ₂ = s.σ₂
  d.σ₃ = s.σ₃
  return dest
end
