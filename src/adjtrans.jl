export AdjointLinearOperator, TransposeLinearOperator, ConjugateLinearOperator,
       adjoint, transpose, conj

# From julialang:stdlib/LinearAlgebra/src/adjtrans.jl
struct AdjointLinearOperator{T} <: AbstractLinearOperator{T}
  parent :: AbstractLinearOperator{T}
end

struct TransposeLinearOperator{T} <: AbstractLinearOperator{T}
  parent :: AbstractLinearOperator{T}
end

struct ConjugateLinearOperator{T} <: AbstractLinearOperator{T}
  parent :: AbstractLinearOperator{T}
end

adjoint(A :: AbstractLinearOperator) = AdjointLinearOperator(A)
adjoint(A :: AdjointLinearOperator) = A.parent
transpose(A :: AbstractLinearOperator) = TransposeLinearOperator(A)
transpose(A :: TransposeLinearOperator) = A.parent
conj(A :: AbstractLinearOperator) = ConjugateLinearOperator(A)
conj(A :: ConjugateLinearOperator) = A.parent

adjoint(A :: ConjugateLinearOperator) = transpose(A.parent)
adjoint(A :: TransposeLinearOperator) = conj(A.parent)
conj(A :: AdjointLinearOperator) = transpose(A.parent)
conj(A :: TransposeLinearOperator) = adjoint(A.parent)
transpose(A :: AdjointLinearOperator) = conj(A.parent)
transpose(A :: ConjugateLinearOperator) = adjoint(A.parent)

nprod(A::AdjointLinearOperator) = nctprod(A.parent)
ntprod(A::AdjointLinearOperator) = nprod(A.parent)   # transpose(A') = conj(A)
nctprod(A::AdjointLinearOperator) = nprod(A.parent)  # (A')' == A

nprod(A::TransposeLinearOperator) = ntprod(A.parent)
ntprod(A::TransposeLinearOperator) = nprod(A.parent)
nctprod(A::TransposeLinearOperator) = nprod(A.parent)  # (transpose(A))' = conj(A)

for f in [:nprod, :ntprod, :nctprod]
  @eval begin
    $f(A::ConjugateLinearOperator) = $f(A.parent)
  end
end

const AdjTrans = Union{AdjointLinearOperator,TransposeLinearOperator}

size(A :: AdjTrans) = size(A.parent)[[2;1]]
size(A :: AdjTrans, d :: Int) = size(A.parent, 3 - d)
size(A :: ConjugateLinearOperator) = size(A.parent)
size(A :: ConjugateLinearOperator, d :: Int) = size(A.parent, d)

for f in [:hermitian, :ishermitian, :symmetric, :issymmetric]
  @eval begin
    $f(A :: AdjTrans) = $f(A.parent)
    $f(A :: ConjugateLinearOperator) = $f(A.parent)
  end
end

function show(io :: IO, op :: AdjointLinearOperator)
  println(io, "Adjoint of the following LinearOperator:")
  show(io, op.parent)
end

function show(io :: IO, op :: TransposeLinearOperator)
  println(io, "Transpose of the following LinearOperator:")
  show(io, op.parent)
end

function show(io :: IO, op :: ConjugateLinearOperator)
  println(io, "Conjugate of the following LinearOperator:")
  show(io, op.parent)
end

function *(op :: AdjointLinearOperator, v :: AbstractVector)
  length(v) == size(op.parent, 1) || throw(LinearOperatorException("shape mismatch"))
  p = op.parent
  ishermitian(p) && return p * v
  if p.ctprod !== nothing
    increase_nctprod(p)
    return p.ctprod(v)
  end
  tprod = p.tprod
  increment_tprod = true
  if p.tprod === nothing
    if issymmetric(p)
      increment_tprod = false
      tprod = p.prod
    else
      throw(LinearOperatorException("unable to infer conjugate transpose operator"))
    end
  end
  if increment_tprod
    increase_ntprod(p)
  else
    increase_nprod(p)
  end
  return conj.(tprod(conj.(v)))
end

function *(op :: TransposeLinearOperator, v :: AbstractVector)
  length(v) == size(op.parent, 1) || throw(LinearOperatorException("shape mismatch"))
  p = op.parent
  issymmetric(p) && return p * v
  if p.tprod !== nothing
    increase_ntprod(p)
    return p.tprod(v)
  end
  increment_ctprod = true
  ctprod = p.ctprod
  if p.ctprod === nothing
    if ishermitian(p)
      increment_ctprod = false
      ctprod = p.prod
    else
      throw(LinearOperatorException("unable to infer transpose operator"))
    end
  end
  if increment_ctprod
    increase_nctprod(p)
  else
    increase_nprod(p)
  end
  return conj.(ctprod(conj.(v)))
end

function *(op :: ConjugateLinearOperator, v :: AbstractVector)
  p = op.parent
  return conj.(p * conj.(v))
end

-(op :: AdjointLinearOperator)   = adjoint(-op.parent)
-(op :: TransposeLinearOperator) = transpose(-op.parent)
-(op :: ConjugateLinearOperator) = conj(-op.parent)

*(op :: AdjointLinearOperator, x :: Number)   = adjoint(op.parent * conj(x))
*(op :: TransposeLinearOperator, x :: Number) = transpose(op.parent * x)
*(op :: ConjugateLinearOperator, x :: Number) = conj(op.parent * conj(x))

*(x :: Number, op :: AdjointLinearOperator)   = adjoint(conj(x) * op.parent)
*(x :: Number, op :: TransposeLinearOperator) = transpose(x * op.parent)
*(x :: Number, op :: ConjugateLinearOperator) = conj(conj(x) * op.parent)
