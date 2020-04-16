using TimerOutputs

export TimedLinearOperator

mutable struct TimedLinearOperator{T} <: AbstractLinearOperator{T}
  timer :: TimerOutput
  op :: AbstractLinearOperator{T}
  prod
  tprod
  ctprod
end

"""
    TimedLinearOperator(op)

Creates a linear operator instrumented with timers from TimerOutputs.
"""
function TimedLinearOperator(op::AbstractLinearOperator{T}) where T
  timer = TimerOutput()
  prod(x) = @timeit timer "prod" op.prod(x)
  tprod(x) = @timeit timer "tprod" op.tprod(x)
  ctprod(x) = @timeit timer "ctprod" op.ctprod(x)
  TimedLinearOperator{T}(timer, op, prod, tprod, ctprod)
end

TimedLinearOperator(op::AdjointLinearOperator) = adjoint(TimedLinearOperator(op.parent))
TimedLinearOperator(op::TransposeLinearOperator) = transpose(TimedLinearOperator(op.parent))
TimedLinearOperator(op::ConjugateLinearOperator) = conj(TimedLinearOperator(op.parent))

for fn ∈ (:size, :shape, :symmetric, :issymmetric, :hermitian, :ishermitian,
          :nprod, :ntprod, :nctprod, :increase_nprod, :increase_ntprod, :increase_nctprod, :reset!)
  @eval begin
    $fn(A::TimedLinearOperator) = $fn(A.op)
  end
end

function show(io :: IO, op :: TimedLinearOperator)
  show(io, op.op)
  show(io, op.timer)
end

