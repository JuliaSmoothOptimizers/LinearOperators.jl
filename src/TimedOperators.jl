using TimerOutputs

export TimedLinearOperator

mutable struct TimedLinearOperator{T, OP <: AbstractLinearOperator{T}, F, Ft, Fct} <:
               AbstractLinearOperator{T}
  timer::TimerOutput
  op::OP
  prod!::F
  tprod!::Ft
  ctprod!::Fct
end

"""
    TimedLinearOperator(op)
Creates a linear operator instrumented with timers from TimerOutputs.
"""
function TimedLinearOperator(op::AbstractLinearOperator{T}) where {T}
  timer = TimerOutput()
  prod!(res, x, α, β) = @timeit timer "prod" op.prod!(res, x, α, β)
  tprod!(res, x, α, β) = @timeit timer "tprod" op.tprod!(res, x, α, β)
  ctprod!(res, x, α, β) = @timeit timer "ctprod" op.ctprod!(res, x, α, β)
  TimedLinearOperator(timer, op, prod!, tprod!, ctprod!)
end

TimedLinearOperator(op::AdjointLinearOperator) = adjoint(TimedLinearOperator(op.parent))
TimedLinearOperator(op::TransposeLinearOperator) = transpose(TimedLinearOperator(op.parent))
TimedLinearOperator(op::ConjugateLinearOperator) = conj(TimedLinearOperator(op.parent))

for fn ∈ (
  :size,
  :shape,
  :issymmetric,
  :ishermitian,
  :has_args5,
  :use_prod5!,
  :isallocated5,
  :allocate_vectors_args3!,
  :nprod,
  :ntprod,
  :nctprod,
  :storage_type,
  :increase_nprod,
  :increase_ntprod,
  :increase_nctprod,
  :reset!,
)
  @eval begin
    $fn(A::TimedLinearOperator) = $fn(A.op)
  end
end

function show(io::IO, op::TimedLinearOperator)
  show(io, op.op)
  show(io, op.timer)
end
