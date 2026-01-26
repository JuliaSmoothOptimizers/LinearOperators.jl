export ShiftedOperator

"A data type to hold information relative to Shifted operators."
mutable struct ShiftedData{T, OpH}
  H::OpH
  σ::T
  function ShiftedData{T, OpH}(H::OpH, σ::T) where {T, OpH}
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("Operator H must be square."))
    new{T, OpH}(H, σ)
  end
end

ShiftedData(H::OpH, σ::T) where {T, OpH} = ShiftedData{T, OpH}(H, σ)

# Forward product: y = α(H + σI)x + βy
function shifted_prod!(y, data::ShiftedData, x, α, β)
  # y = α * H * x + β * y
  mul!(y, data.H, x, α, β)

  # y = y + (α * σ) * x
  if !(iszero(data.σ) && iszero(α))
    axpy!(α * data.σ, x, y)
  end
  return y
end

# Transpose product: y = α(Hᵀ + σI)x + βy
function shifted_tprod!(y, data::ShiftedData, x, α, β)
  # y = α * Hᵀ * x + β * y
  mul!(y, transpose(data.H), x, α, β)

  # y = y + (α * σ) * x
  if !iszero(data.σ)
    axpy!(α * data.σ, x, y)
  end
  return y
end

# Conjugate Transpose (Adjoint) product: y = α(Hᴴ + conj(σ)I)x + βy
function shifted_ctprod!(y, data::ShiftedData, x, α, β)
  # y = α * Hᴴ * x + β * y
  mul!(y, adjoint(data.H), x, α, β)

  # y = y + (α * conj(σ)) * x
  if !(iszero(data.σ) && iszero(α))
    axpy!(α * conj(data.σ), x, y)
  end
  return y
end

"""
    ShiftedOperator(H, σ=0)

Constructs a linear operator representing `op = H + σI`.

"""
mutable struct ShiftedOperator{T, OpH, F, Ft, Fct} <: AbstractLinearOperator{T}
  nrow::Int
  ncol::Int
  symmetric::Bool
  hermitian::Bool
  prod!::F      # Closure for op * x
  tprod!::Ft    # Closure for transpose(op) * x
  ctprod!::Fct  # Closure for adjoint(op) * x
  data::ShiftedData{T, OpH}
  nprod::Int    # Internal counter for products
  ntprod::Int   # Internal counter for transpose products
  nctprod::Int  # Internal counter for adjoint products
end

# Internal Constructor
function ShiftedOperator(
  nrow::Int,
  ncol::Int,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  data::ShiftedData{T, OpH},
) where {T, OpH, F, Ft, Fct}
  return ShiftedOperator{T, OpH, F, Ft, Fct}(
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
end

function ShiftedOperator(H::OpH, σ_in::Number = zero(eltype(H))) where {OpH}
  T = promote_type(eltype(H), typeof(σ_in))

  data = ShiftedData(H, σ_in)
  σ = convert(T, σ_in)

  prod! = (y, x, α, β) -> shifted_prod!(y, data, x, α, β)
  tprod! = (y, x, α, β) -> shifted_tprod!(y, data, x, α, β)
  ctprod! = (y, x, α, β) -> shifted_ctprod!(y, data, x, α, β)

  n = size(H, 1)

  is_sym = issymmetric(H)
  is_herm = ishermitian(H) && isreal(σ)

  return ShiftedOperator(n, n, is_sym, is_herm, prod!, tprod!, ctprod!, data)
end

size(op::ShiftedOperator) = (op.nrow, op.ncol)
issymmetric(op::ShiftedOperator) = op.symmetric
ishermitian(op::ShiftedOperator) = op.hermitian

has_args5(op::ShiftedOperator) = true
use_prod5!(op::ShiftedOperator) = true

isallocated5(op::ShiftedOperator) = true

storage_type(op::ShiftedOperator{T}) where {T} = Vector{T}

function transpose(op::ShiftedOperator)
  # (H + σI)ᵀ = Hᵀ + σI
  return ShiftedOperator(transpose(op.data.H), op.data.σ)
end

function adjoint(op::ShiftedOperator)
  # (H + σI)ᴴ = Hᴴ + conj(σ)I
  return ShiftedOperator(adjoint(op.data.H), conj(op.data.σ))
end

function reset!(op::ShiftedOperator)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
