module LinearOperatorsChainRulesCoreExt

using LinearOperators
isdefined(Base, :get_extension) ? (import ChainRulesCore) : (import ..ChainRulesCore)

function ChainRulesCore.frule((_, Δx, _), ::typeof(*), op::AbstractLinearOperator{T}, x::AbstractVector{S}) where {T, S}
  y = op*x
  Δy = op*Δx
  return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), op::AbstractLinearOperator{T}, x::AbstractVector{S}) where {T, S}
  y = op*x
  project_x = ChainRulesCore.ProjectTo(x)
  function mul_pullback(ȳ)
      x̄ = project_x( adjoint(op)*ChainRulesCore.unthunk(ȳ) )
      return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
  end
  return y, mul_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(*), x::Union{LinearOperators.Adjoint{S, V}, LinearOperators.Transpose{S, V} }, op::AbstractLinearOperator{T}) where {T, S, V <: AbstractVector{S}}
  y = x*op
  Δy = Δx*op
  return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), x::LinearOperators.Transpose{S, V}, op::AbstractLinearOperator{T}) where {T, S, V <: AbstractVector{S}}
  y = x*op
  project_x = ChainRulesCore.ProjectTo(x)
  function mul_pullback(ȳ)
      # needed to make sure that ȳ is recognized as Transposed
      # ȳ_ = transpose(collect(vec(ChainRulesCore.unthunk(ȳ))))
      ȳ_ = transpose(vec(ChainRulesCore.unthunk(ȳ)))
      x̄ = project_x(ȳ_*adjoint(op))
      return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
  end
  return y, mul_pullback
end
function ChainRulesCore.rrule(::typeof(*), x::LinearOperators.Adjoint{S, V}, op::AbstractLinearOperator{T}) where {T, S, V <: AbstractVector{S}}
  y = x*op
  project_x = ChainRulesCore.ProjectTo(x)
  function mul_pullback(ȳ)
      # needed to make sure that ȳ is recognized as Adjoint
      # ȳ_ = adjoint(collect(vec(ChainRulesCore.unthunk(ȳ))))
      ȳ_ = adjoint(conj.(vec(ChainRulesCore.unthunk(ȳ))))
      x̄ = project_x(ȳ_*adjoint(op))
      return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
  end
  return y, mul_pullback
end

end # module