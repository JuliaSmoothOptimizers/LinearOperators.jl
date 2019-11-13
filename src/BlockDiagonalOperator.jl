import Base.eltypeof, Base.promote_eltypeof

eltypeof(op::AbstractLinearOperator) = eltype(op)  # need this for promote_eltypeof

export BlockDiagonalOperator

function BlockDiagonalOperator(ops...)
  nrow = ncol = 0
  for op ∈ ops
    m, n = size(op)
    nrow += m
    ncol += n
  end
  T = promote_eltypeof(ops...)

  function prod(x)
    y = zeros(T, nrow)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      y[k + 1 : k + m] .= op * x[j + 1 : j + n]
      k += m
      j += n
    end
    y
  end

  function tprod(x)
    y = zeros(T, ncol)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      y[k + 1 : k + n] .= transpose(op) * x[j + 1 : j + m]
      k += n
      j += m
    end
    y
  end

  function ctprod(x)
    y = zeros(T, ncol)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      y[k + 1 : k + n] .= op' * x[j + 1 : j + m]
      k += n
      j += m
    end
    y
  end

  symmetric = all((issymmetric(op) for op ∈ ops))
  hermitian = all((ishermitian(op) for op ∈ ops))

  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

