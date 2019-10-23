export AdjointLinearOperator, TransposeLinearOperator, adjoint, transpose

# From julialang:stdlib/LinearAlgebra/src/adjtrans.jl
struct AdjointLinearOperator{T,F1,F2,F3} <: AbstractLinearOperator{T,F1,F2,F3}
  parent :: AbstractLinearOperator{T,F1,F2,F3}
end

struct TransposeLinearOperator{T,F1,F2,F3} <: AbstractLinearOperator{T,F1,F2,F3}
  parent :: AbstractLinearOperator{T,F1,F2,F3}
end

struct ConjugateLinearOperator{T,F1,F2,F3} <: AbstractLinearOperator{T,F1,F2,F3}
  parent :: AbstractLinearOperator{T,F1,F2,F3}
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

const AdjTrans = Union{AdjointLinearOperator,TransposeLinearOperator}

size(A :: AdjTrans) = size(A.parent)[[2;1]]
size(A :: AdjTrans, d :: Int) = size(A.parent, 3 - d)
size(A :: ConjugateLinearOperator) = size(A.parent)
size(A :: ConjugateLinearOperator, d :: Int) = size(A.parent, d)

for f in [:hermitian, :ishermitian, :symmetric, :issymmetric]
  @eval begin
    $f(A :: AdjTrans) = $f(A.parent)
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
  p = op.parent
  ishermitian(p) && return p * v
  p.ctprod !== nothing && return p.ctprod(v)
  tprod = p.tprod
  if p.tprod === nothing
    if issymmetric(p)
      tprod = p.prod
    else
      throw(LinearOperatorException("unable to infer conjugate transpose operator"))
    end
  end
  return conj.(tprod(conj.(v)))
end

function *(op :: TransposeLinearOperator, v :: AbstractVector)
  p = op.parent
  issymmetric(p) && return p * v
  p.tprod !== nothing && return p.tprod(v)
  ctprod = p.ctprod
  if p.ctprod === nothing
    if ishermitian(p)
      ctprod = p.prod
    else
      throw(LinearOperatorException("unable to infer transpose operator"))
    end
  end
  return conj.(ctprod(conj.(v)))
end

function *(op :: ConjugateLinearOperator, v :: AbstractVector)
  p = op.parent
  return conj.(p * v)
end
