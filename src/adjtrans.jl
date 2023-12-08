import Base.transpose, Base.adjoint, Base.conj

export AdjointLinearOperator,
  TransposeLinearOperator, ConjugateLinearOperator, adjoint, transpose, conj

# From julialang:stdlib/LinearAlgebra/src/adjtrans.jl
struct AdjointLinearOperator{T, S} <: AbstractLinearOperator{T}
  parent::S
  function AdjointLinearOperator{T, S}(A::S) where {T, S}
    new(A)
  end
end

struct TransposeLinearOperator{T, S} <: AbstractLinearOperator{T}
  parent::S
  function TransposeLinearOperator{T, S}(A::S) where {T, S}
    new(A)
  end
end

struct ConjugateLinearOperator{T, S} <: AbstractLinearOperator{T}
  parent::S
  function ConjugateLinearOperator{T, S}(A::S) where {T, S}
    new(A)
  end
end

AdjointLinearOperator(A) = AdjointLinearOperator{eltype(A), typeof(A)}(A)
TransposeLinearOperator(A) = TransposeLinearOperator{eltype(A), typeof(A)}(A)
ConjugateLinearOperator(A) = ConjugateLinearOperator{eltype(A), typeof(A)}(A)

adjoint(A::AbstractLinearOperator) = AdjointLinearOperator(A)
adjoint(A::AdjointLinearOperator) = A.parent
transpose(A::AbstractLinearOperator) = TransposeLinearOperator(A)
transpose(A::TransposeLinearOperator) = A.parent
conj(A::AbstractLinearOperator) = ConjugateLinearOperator(A)
conj(A::ConjugateLinearOperator) = A.parent

adjoint(A::ConjugateLinearOperator) = transpose(A.parent)
adjoint(A::TransposeLinearOperator) = conj(A.parent)
conj(A::AdjointLinearOperator) = transpose(A.parent)
conj(A::TransposeLinearOperator) = adjoint(A.parent)
transpose(A::AdjointLinearOperator) = conj(A.parent)
transpose(A::ConjugateLinearOperator) = adjoint(A.parent)

nprod(A::AdjointLinearOperator) = nctprod(A.parent)
ntprod(A::AdjointLinearOperator) = nprod(A.parent)   # transpose(A') = conj(A)
nctprod(A::AdjointLinearOperator) = nprod(A.parent)  # (A')' == A

nprod(A::TransposeLinearOperator) = ntprod(A.parent)
ntprod(A::TransposeLinearOperator) = nprod(A.parent)
nctprod(A::TransposeLinearOperator) = nprod(A.parent)  # (transpose(A))' = conj(A)

for f in [:nprod, :ntprod, :nctprod, :increase_nprod!, :increase_ntprod!, :increase_nctprod!]
  @eval begin
    $f(A::ConjugateLinearOperator) = $f(A.parent)
  end
end

const AdjTrans = Union{AdjointLinearOperator, TransposeLinearOperator}

size(A::AdjTrans) = size(A.parent)[[2; 1]]
size(A::AdjTrans, d::Int) = size(A.parent, 3 - d)
size(A::ConjugateLinearOperator) = size(A.parent)
size(A::ConjugateLinearOperator, d::Int) = size(A.parent, d)

for f in [
  :ishermitian,
  :issymmetric,
  :has_args5,
  :use_prod5!,
  :isallocated5,
  :allocate_vectors_args3!,
  :storage_type,
]
  @eval begin
    $f(A::AdjTrans) = $f(A.parent)
    $f(A::ConjugateLinearOperator) = $f(A.parent)
  end
end

function show(io::IO, op::AdjointLinearOperator)
  println(io, "Adjoint of the following LinearOperator:")
  show(io, op.parent)
end

function show(io::IO, op::TransposeLinearOperator)
  println(io, "Transpose of the following LinearOperator:")
  show(io, op.parent)
end

function show(io::IO, op::ConjugateLinearOperator)
  println(io, "Conjugate of the following LinearOperator:")
  show(io, op.parent)
end

function mul!(
  res::AbstractVector,
  op::AdjointLinearOperator{T, S},
  v::AbstractVector,
  α,
  β,
) where {T, S}
  p = op.parent
  use_p5! = use_prod5!(p)
  has_args5(op) || (β == 0) || isallocated5(op) || allocate_vectors_args3!(op)
  (length(v) == size(p, 1) && length(res) == size(p, 2)) ||
    throw(LinearOperatorException("shape mismatch"))
  if ishermitian(p)
    return mul!(res, p, v, α, β)
  end
  if p.ctprod! !== nothing
    increase_nctprod!(p)
    if use_p5!
      return p.ctprod!(res, v, α, β)
    else
      return prod3!(res, p.ctprod!, v, α, β, p.Mtu5)
    end
  end
  tprod! = p.tprod!
  increment_tprod = true
  if p.tprod! === nothing
    if issymmetric(p)
      increment_tprod = false
      tprod! = p.prod!
    else
      throw(LinearOperatorException("unable to infer conjugate transpose operator"))
    end
  end
  if increment_tprod
    increase_ntprod!(p)
  else
    increase_nprod!(p)
  end
  if use_p5!
    tprod!(res, v, α, β)
  else
    prod3!(res, tprod!, v, α, β, p.Mtu5)
  end
  conj!(res)
end

function mul!(
  res::AbstractVector,
  op::TransposeLinearOperator{T, S},
  v::AbstractVector,
  α,
  β,
) where {T, S}
  p = op.parent
  use_p5! = use_prod5!(p)
  has_args5(op) || (α == 1 && β == 0) || isallocated5(op) || allocate_vectors_args3!(op)
  (length(v) == size(p, 1) && length(res) == size(p, 2)) ||
    throw(LinearOperatorException("shape mismatch"))
  if issymmetric(p)
    return mul!(res, p, v, α, β)
  end
  if p.tprod! !== nothing
    increase_ntprod!(p)
    if use_p5!
      return p.tprod!(res, v, α, β)
    else
      return prod3!(res, p.tprod!, v, α, β, p.Mtu5)
    end
  end
  increment_ctprod = true
  ctprod! = p.ctprod!
  if p.ctprod! === nothing
    if ishermitian(p)
      increment_ctprod = false
      ctprod! = p.prod!
    else
      throw(LinearOperatorException("unable to infer transpose operator"))
    end
  end
  if increment_ctprod
    increase_nctprod!(p)
  else
    increase_nprod!(p)
  end
  conj!(res)
  if use_p5!
    ctprod!(res, conj.(v), conj(α), conj(β))
  else
    prod3!(res, ctprod!, conj.(v), conj(α), conj(β), p.Mtu5)
  end
  conj!(res)
end

function mul!(
  res::AbstractVector,
  op::ConjugateLinearOperator{T, S},
  v::AbstractVector,
  α,
  β,
) where {T, S}
  p = op.parent
  mul!(res, p, conj.(v), α, β)
  conj!(res)
end

-(op::AdjointLinearOperator) = adjoint(-op.parent)
-(op::TransposeLinearOperator) = transpose(-op.parent)
-(op::ConjugateLinearOperator) = conj(-op.parent)

*(op::AdjointLinearOperator, x::Number) = adjoint(op.parent * conj(x))
*(op::TransposeLinearOperator, x::Number) = transpose(op.parent * x)
*(op::ConjugateLinearOperator, x::Number) = conj(op.parent * conj(x))

*(x::Number, op::AdjointLinearOperator) = adjoint(conj(x) * op.parent)
*(x::Number, op::TransposeLinearOperator) = transpose(x * op.parent)
*(x::Number, op::ConjugateLinearOperator) = conj(conj(x) * op.parent)
