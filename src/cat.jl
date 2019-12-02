import Base.hcat, Base.vcat, Base.hvcat

hcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = hcat(A, LinearOperator(B))

hcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = hcat(LinearOperator(A), B)

function hcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  size(A, 1) == size(B, 1) || throw(LinearOperatorException("hcat: inconsistent row sizes"))

  nrow  = size(A, 1)
  Ancol, Bncol = size(A, 2), size(B, 2)
  ncol  = Ancol + Bncol
  S = promote_type(eltype(A), eltype(B))

  prod = @closure v -> A * v[1:Ancol] + B * v[Ancol+1:length(v)]
  tprod  = @closure v -> [transpose(A) * v; transpose(B) * v;]
  ctprod = @closure v -> [A' * v; B' * v;]
  LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function hcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end


vcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = vcat(A, LinearOperator(B))

vcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = vcat(LinearOperator(A), B)

function vcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  size(A, 2) == size(B, 2) || throw(LinearOperatorException("vcat: inconsistent column sizes"))

  Anrow, Bnrow = size(A, 1), size(B, 1)
  nrow  = Anrow + Bnrow
  ncol  = size(A, 2)
  S = promote_type(eltype(A), eltype(B))

  prod = @closure v -> [A * v; B * v;]
  tprod = @closure v -> transpose(A) * v +  transpose(B) * v
  ctprod = @closure v -> A' * v[1:Anrow] + B' * v[Anrow+1:length(v)]
  return LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function vcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end

# Removed by https://github.com/JuliaLang/julia/pull/24017
function hvcat(rows :: Tuple{Vararg{Int}}, ops :: AbstractLinearOperator...)
  nbr = length(rows)
  rs = Array{AbstractLinearOperator,1}(undef, nbr)
  a = 1
  for i = 1:nbr
    rs[i] = hcat(ops[a:a-1+rows[i]]...)
    a += rows[i]
  end
  vcat(rs...)
end
