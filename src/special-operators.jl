import Base.eltypeof, Base.promote_eltypeof

export opEye, opOnes, opZeros, opDiagonal, opRestriction, opExtension, BlockDiagonalOperator

"""`opEye()`

Identity operator.
```
opI = opEye()
v = rand(5)
@assert opI * v === v
```
"""
struct opEye <: AbstractLinearOperator{Any} end

*(::opEye, x::AbstractArray{T, 1} where {T}) = x
*(x::AbstractArray{T, 1} where {T}, ::opEye) = x
*(::opEye, A::AbstractArray{T, 2} where {T}) = A
*(A::AbstractArray{T, 2} where {T}, ::opEye) = A
*(::opEye, T::AbstractLinearOperator) = T
*(T::AbstractLinearOperator, ::opEye) = T
*(::opEye, T::opEye) = T

function show(io::IO, op::opEye)
  println(io, "Identity operator")
end

"""
    opEye(T, n)
    opEye(n)

Identity operator of order `n` and of data type `T` (defaults to `Float64`).
"""
function opEye(T::DataType, n::Int)
  prod = @closure v -> copy(v)
  LinearOperator{T}(n, n, true, true, prod, prod, prod)
end

opEye(n::Int) = opEye(Float64, n)

# TODO: not type stable
"""
    opEye(T, nrow, ncol)
    opEye(nrow, ncol)

Rectangular identity operator of size `nrow`x`ncol` and of data type `T`
(defaults to `Float64`).
"""
function opEye(T::DataType, nrow::Int, ncol::Int)
  if nrow == ncol
    return opEye(T, nrow)
  end
  if nrow > ncol
    prod = @closure v -> [v; zeros(T, nrow - ncol)]
    tprod = @closure v -> v[1:ncol]
  else
    prod = @closure v -> v[1:nrow]
    tprod = @closure v -> [v; zeros(T, ncol - nrow)]
  end
  return LinearOperator{T}(nrow, ncol, false, false, prod, tprod, tprod)
end

opEye(nrow::Int, ncol::Int) = opEye(Float64, nrow, ncol)

"""
    opOnes(T, nrow, ncol)
    opOnes(nrow, ncol)

Operator of all ones of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
function opOnes(T::DataType, nrow::Int, ncol::Int)
  prod = @closure v -> sum(v) * ones(T, nrow)
  tprod = @closure u -> sum(u) * ones(T, ncol)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod, tprod, tprod)
end

opOnes(nrow::Int, ncol::Int) = opOnes(Float64, nrow, ncol)

"""
    opZeros(T, nrow, ncol)
    opZeros(nrow, ncol)

Zero operator of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
function opZeros(T::DataType, nrow::Int, ncol::Int)
  prod = @closure v -> zeros(promote_type(T, eltype(v)), nrow)
  tprod = @closure u -> zeros(promote_type(T, eltype(u)), ncol)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod, tprod, tprod)
end

opZeros(nrow::Int, ncol::Int) = opZeros(Float64, nrow, ncol)

"""
    opDiagonal(d)

Diagonal operator with the vector `d` on its main diagonal.
"""
function opDiagonal(d::AbstractVector{T}) where {T}
  prod = @closure v -> v .* d
  ctprod = @closure w -> w .* conj(d)
  LinearOperator{T}(length(d), length(d), true, isreal(d), prod, prod, ctprod)
end

#TODO: not type stable
"""
    opDiagonal(nrow, ncol, d)

Rectangular diagonal operator of size `nrow`-by-`ncol` with the vector `d` on
its main diagonal.
"""
function opDiagonal(nrow::Int, ncol::Int, d::AbstractVector{T}) where {T}
  nrow == ncol <= length(d) && (return opDiagonal(d[1:nrow]))
  if nrow > ncol
    prod = @closure v -> [v .* d; zeros(nrow - ncol)]
    tprod = @closure u -> u[1:ncol] .* d
    ctprod = @closure w -> w[1:ncol] .* conj(d)
  else
    prod = @closure v -> v[1:nrow] .* d
    tprod = @closure u -> [u .* d; zeros(ncol - nrow)]
    ctprod = @closure w -> [w .* conj(d); zeros(ncol - nrow)]
  end
  LinearOperator{T}(nrow, ncol, false, false, prod, tprod, ctprod)
end

"""
    Z = opRestriction(I, ncol)
    Z = opRestriction(:, ncol)

Creates a LinearOperator restricting a `ncol`-sized vector to indices `I`.
The operation `Z * v` is equivalent to `v[I]`. `I` can be `:`.

    Z = opRestriction(k, ncol)

Alias for `opRestriction([k], ncol)`.
"""
function opRestriction(I::LinearOperatorIndexType, ncol::Int)
  all(1 .≤ I .≤ ncol) || throw(LinearOperatorException("indices should be between 1 and $ncol"))
  nrow = length(I)
  prod = @closure x -> x[I]
  tprod = @closure x -> begin
    z = zeros(eltype(x), ncol)
    z[I] = x
    return z
  end
  return LinearOperator{Int}(nrow, ncol, false, false, prod, tprod, tprod)
end

opRestriction(::Colon, ncol::Int) = opEye(Int, ncol)

opRestriction(k::Int, ncol::Int) = opRestriction([k], ncol)

"""
    Z = opExtension(I, ncol)
    Z = opExtension(:, ncol)

Creates a LinearOperator extending a vector of size `length(I)` to size `ncol`,
where the position of the elements on the new vector are given by the indices
`I`.
The operation `w = Z * v` is equivalent to `w = zeros(ncol); w[I] = v`.

    Z = opExtension(k, ncol)

Alias for `opExtension([k], ncol)`.
"""
opExtension(I::LinearOperatorIndexType, ncol::Int) = opRestriction(I, ncol)'

opExtension(::Colon, ncol::Int) = opEye(Int, ncol)

opExtension(k::Int, ncol::Int) = opExtension([k], ncol)

# indexing for linear operators
import Base.getindex
function getindex(
  op::AbstractLinearOperator,
  rows::Union{LinearOperatorIndexType, Int, Colon},
  cols::Union{LinearOperatorIndexType, Int, Colon},
)
  R = opRestriction(rows, size(op, 1))
  E = opExtension(cols, size(op, 2))
  return R * op * E
end

eltypeof(op::AbstractLinearOperator) = eltype(op)  # need this for promote_eltypeof

"""
    BlockDiagonalOperator(M1, M2, ..., Mn)

Creates a block-diagonal linear operator:

    [ M1           ]
    [    M2        ]
    [       ...    ]
    [           Mn ]
"""
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
      y[(k + 1):(k + m)] .= op * x[(j + 1):(j + n)]
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
      y[(k + 1):(k + n)] .= transpose(op) * x[(j + 1):(j + m)]
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
      y[(k + 1):(k + n)] .= op' * x[(j + 1):(j + m)]
      k += n
      j += m
    end
    y
  end

  symmetric = all((issymmetric(op) for op ∈ ops))
  hermitian = all((ishermitian(op) for op ∈ ops))

  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end
