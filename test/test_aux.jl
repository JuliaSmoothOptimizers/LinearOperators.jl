using LinearAlgebra, SparseArrays

function simple_matrix(T::Type, nrow::Int, ncol::Int;
                       symmetric::Bool=false)
  symmetric && @assert nrow == ncol
  if nrow == ncol == 1
    return ones(T, nrow, ncol)
  end
  U, _ = qr(rand(T, nrow, nrow))
  local V
  if symmetric
    V = U
  else
    V, _ = qr(rand(T, ncol, ncol))
  end
  S = T[(1 + (i - 1) / (nrow-1)) * (i == j) for i = 1:nrow, j = 1:ncol]
  return U * S * V'
end

function simple_sparse_matrix(T::Type, nrow::Int, ncol::Int;
                              symmetric::Bool=false)
  m = min(nrow, ncol)
  p = floor(Int, sqrt(m))
  q = div(m, p)
  M = kron(Matrix(I, q, q), simple_matrix(T, p, p, symmetric=symmetric))
  pq = p*q
  R = if m > pq
    simple_matrix(T, nrow - pq, ncol - pq, symmetric=symmetric)
  else
    spzeros(nrow - pq, ncol - pq)
  end
  A = [sparse(M) spzeros(pq, ncol-pq); spzeros(nrow-pq, pq) R]
end

simple_vector(T::Type, n::Int) = T[-(-one(T))^i for i = 1:n]

