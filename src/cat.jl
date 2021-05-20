import Base.hcat, Base.vcat, Base.hvcat

hcat(A::AbstractLinearOperator, B::AbstractMatrix) = hcat(A, LinearOperator(B))

hcat(A::AbstractMatrix, B::AbstractLinearOperator) = hcat(LinearOperator(A), B)

function hcat_prod!(res::AbstractVector, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}, 
                    Ancol::I, nV::I, v::AbstractVector, α, β) where {T,I<:Integer}
  mul!(res, A, view(v, 1:Ancol), α, β)
  mul!(res, B, view(v, (Ancol+1): nV), α, one(T))
end

function hcat_ctprod!(res::AbstractVector, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T},
                     Ancol::I, nV::I, u::AbstractVector, α, β) where {T,I<:Integer}
  mul!(view(res, 1:Ancol), A, u, α, β)
  mul!(view(res, (Ancol+1): nV), B, u, α, β)
end
                     
function hcat(A::AbstractLinearOperator, B::AbstractLinearOperator)
  size(A, 1) == size(B, 1) || throw(LinearOperatorException("hcat: inconsistent row sizes"))

  nrow = size(A, 1)
  Ancol, Bncol = size(A, 2), size(B, 2)
  ncol = Ancol + Bncol
  S = promote_type(eltype(A), eltype(B))
  if typeof(A) <: AdjointLinearOperator || typeof(A) <: TransposeLinearOperator || typeof(A) <: ConjugateLinearOperator
    vtmp = A.parent.Mv 
  else
    vtmp = A.Mv
  end
  MvAB = similar(vtmp, S, nrow)
  MtuAB = similar(MvAB, S, ncol)
  MawAB = similar(MvAB, S, ncol)

  prod = @closure (res, v, α, β) -> hcat_prod!(res, A, B, Ancol, Ancol+Bncol, v, α, β)
  tprod = @closure (res, u, α, β) -> hcat_ctprod!(res, transpose(A), transpose(B), Ancol, Ancol+Bncol, u, α, β)
  ctprod = @closure (res, w, α, β) -> hcat_ctprod!(res, adjoint(A), adjoint(B), Ancol, Ancol+Bncol, w, α, β)
  LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod, MvAB, MtuAB, MawAB)
end

function hcat(ops::AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end

vcat(A::AbstractLinearOperator, B::AbstractMatrix) = vcat(A, LinearOperator(B))

vcat(A::AbstractMatrix, B::AbstractLinearOperator) = vcat(LinearOperator(A), B)

function vcat_prod!(res::AbstractVector, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T},
                    Anrow::I, nV::I, u::AbstractVector, α, β) where {T,I<:Integer}
  mul!(view(res, 1:Anrow), A, u, α, β)
  mul!(view(res, (Anrow+1): nV), B, u, α, β)
end

function vcat_ctprod!(res::AbstractVector, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}, 
                     Anrow::I, nV::I, v::AbstractVector, α, β) where {T,I<:Integer}
  mul!(res, A, view(v, 1:Anrow), α, β)
  mul!(res, B, view(v, (Anrow+1): nV), α, one(T))
end

function vcat(A::AbstractLinearOperator, B::AbstractLinearOperator)
  size(A, 2) == size(B, 2) || throw(LinearOperatorException("vcat: inconsistent column sizes"))

  Anrow, Bnrow = size(A, 1), size(B, 1)
  nrow = Anrow + Bnrow
  ncol = size(A, 2)
  S = promote_type(eltype(A), eltype(B))
  if typeof(A) <: AdjointLinearOperator || typeof(A) <: TransposeLinearOperator || typeof(A) <: ConjugateLinearOperator
    vtmp = A.parent.Mv 
  else
    vtmp = A.Mv
  end
  MvAB = similar(vtmp, S, nrow)
  MtuAB = similar(MvAB, S, ncol)
  MawAB = similar(MvAB, S, ncol)

  prod! = @closure (res, v, α, β) -> vcat_prod!(res, A, B, Anrow, Anrow+Bnrow, v, α, β)
  tprod! = @closure (res, u, α, β) -> vcat_ctprod!(res, transpose(A), transpose(B), Anrow, Anrow+Bnrow, u, α, β)
  ctprod! = @closure (res, w, α, β) -> vcat_ctprod!(res, adjoint(A), adjoint(B), Anrow, Anrow+Bnrow, w, α, β)
  return LinearOperator{S}(nrow, ncol, false, false, prod!, tprod!, ctprod!, MvAB, MtuAB, MawAB)
end

function vcat(ops::AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end

# Removed by https://github.com/JuliaLang/julia/pull/24017
function hvcat(rows::Tuple{Vararg{Int}}, ops::AbstractLinearOperator...)
  nbr = length(rows)
  rs = Array{AbstractLinearOperator, 1}(undef, nbr)
  a = 1
  for i = 1:nbr
    rs[i] = hcat(ops[a:(a - 1 + rows[i])]...)
    a += rows[i]
  end
  vcat(rs...)
end
