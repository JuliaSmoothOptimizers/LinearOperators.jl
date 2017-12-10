export opKron

function opKron(A :: AbstractMatrix,
                B :: Union{AbstractMatrix, AbstractLinearOperator})
  m, n = size(A)
  p, q = size(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    S = promote_type(T, eltype(x))
    opx = zeros(S, m * p)
    Bx = zeros(S, p)
    for j = 1:n
      @views Bx .= B * x[(j - 1) * q + (1:q)]
      for i = 1:m
        aij = A[i,j]
        if aij != 0
          @views opx[(i - 1) * p + (1:p)] .+= aij * Bx
        end
      end
    end
    return opx
  end
  function tprod(x)
    S = promote_type(T, eltype(x))
    opx = zeros(S, n * q)
    Btx = zeros(S, q)
    for i = 1:m
      @views Btx .= B.' * x[(i - 1) * p + (1:p)]
      for j = 1:n
        aij = A[i,j]
        if aij != 0
          @views opx[(j - 1) * q + (1:q)] .+= aij * Btx
        end
      end
    end
    return opx
  end
  function ctprod(x)
    S = promote_type(T, eltype(x))
    opx = zeros(S, n * q)
    Btx = zeros(S, q)
    for i = 1:m
      @views Btx .= B' * x[(i - 1) * p + (1:p)]
      for j = 1:n
        aij = A[i,j]'
        if aij != 0
          @views opx[(j - 1) * q + (1:q)] .+= aij * Btx
        end
      end
    end
    return opx
  end
  symm = issymmetric(A) && issymmetric(B)
  herm = ishermitian(A) && ishermitian(B)
  return LinearOperator{T}(m * p, n * q, symm, herm, prod, tprod, ctprod)
end
