import Base.eltypeof, Base.promote_eltypeof, LinearAlgebra.lmul!, LinearAlgebra.rmul!

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

function mulOpEye!(res, v, α, β::T, n_min) where T
  if β == zero(T)
    res[1:n_min] .= @views α .* v[1:n_min]
    res[n_min+1:end] .= 0 
  else
    res[1:n_min] .= @views α .* v[1:n_min] .+ β .* res[1:n_min]
    res[n_min+1:end] .= β
  end
end

"""
    opEye(T, n)
    opEye(n)

Identity operator of order `n` and of data type `T` (defaults to `Float64`).
"""
function opEye(T::DataType, n::Int)
  prod! = @closure (res, v, α, β) -> mulOpEye!(res, v, α, β, n)
  LinearOperator{T}(n, n, true, true, prod!, prod!, prod!)
end

opEye(n::Int) = opEye(Float64, n)

# TODO: not type stable
"""
    opEye(T, nrow, ncol)
    opEye(nrow, ncol)

Rectangular identity operator of size `nrow`x`ncol` and of data type `T`
(defaults to `Float64`).
"""
function opEye(T::DataType, nrow::I, ncol::I) where {I<:Integer}
  if nrow == ncol
    return opEye(T, nrow)
  end
  prod! = @closure (res, v, α, β) -> mulOpEye!(res, v, α, β, min(nrow, ncol))
  return LinearOperator{T}(nrow, ncol, false, false, prod!, prod!, prod!)
end

opEye(nrow::I, ncol::I) where {I<:Integer} = opEye(Float64, nrow, ncol)

function mulOpOnes!(res, v, α, β::T) where T
  if β == zero(T)
    res .= (α * sum(v))
  else
    res .= (α * sum(v)) .+ β .* res
  end
end

"""
    opOnes(T, nrow, ncol)
    opOnes(nrow, ncol)

Operator of all ones of size `nrow`-by-`ncol` of data type `T` (defaults to
`Float64`).
"""
function opOnes(T::DataType, nrow::I, ncol::I) where {I<:Integer}
  prod! = @closure (res, v, α, β) -> mulOpOnes!(res, v, α, β)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod!, prod!, prod!)
end

opOnes(nrow::I, ncol::I) where {I<:Integer} = opOnes(Float64, nrow, ncol)

function mulOpZeros!(res, v, α, β::T) where T
  if β == zero(T)
    res .= 0
  else
    res .*= β
  end 
end

"""
    opZeros(T, nrow, ncol)
    opZeros(nrow, ncol)

Zero operator of size `nrow`-by-`ncol`, of data type `T` (defaults to
`Float64`).
"""
function opZeros(T::DataType, nrow::I, ncol::I) where {I<:Integer}
  prod! = @closure (res, v, α, β) -> mulOpZeros!(res, v, α, β)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod!, prod!, prod!)
end

opZeros(nrow::I, ncol::I) where {I<:Integer} = opZeros(Float64, nrow, ncol)

function mulSquareOpDiagonal!(res, d, v, α, β::T) where T
  if β == zero(T)
    res .= α .* d .* v
  else 
    res .= α .* d .* v .+ β .* res
  end
end

"""
    opDiagonal(d)

Diagonal operator with the vector `d` on its main diagonal.
"""
function opDiagonal(d::AbstractVector{T}) where {T}
  prod! = @closure (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β)
  ctprod! = @closure (res, w, α, β) -> mulSquareOpDiagonal!(res, conj.(d), w, α, β)
  LinearOperator{T}(length(d), length(d), true, isreal(d), prod!, prod!, ctprod!)
end

#TODO: not type stable
function mulOpDiagonal!(res, d, v, α, β::T, n_min) where T
  if β == zero(T)
    res[1:n_min] .= @views α .* d[1:n_min] .* v[1:n_min]
  else
    res[1:n_min] .= @views α .* d[1:n_min] .* v[1:n_min] .+ β .* res[1:n_min]
  end
  res[n_min+1:end] .= 0
end
"""
    opDiagonal(nrow, ncol, d)

Rectangular diagonal operator of size `nrow`-by-`ncol` with the vector `d` on
its main diagonal and storage vectors `Mv`, `Mtu`, `Maw`.
"""
function opDiagonal(nrow::I, ncol::I, d::AbstractVector{T}) where {T,I<:Integer}
  nrow == ncol <= length(d) && (return opDiagonal(d[1:nrow]))
  n_min = min(nrow, ncol)
  prod! = @closure (res, v, α, β) -> mulOpDiagonal!(res, d, v, α, β, n_min)
  tprod! = @closure (res, u, α, β) -> mulOpDiagonal!(res, d, u, α, β, n_min)
  ctprod! = @closure (res, w, α, β) -> mulOpDiagonal!(res, conj.(d), w, α, β, n_min)
  LinearOperator{T}(nrow, ncol, false, false, prod!, tprod!, ctprod!)
end

function mulRestrict!(res, I, v, α, β)
  res .= v[I]
end

function multRestrict!(res, I, u, α, β)
  res .= 0
  res[I] = u
end
  
"""
    Z = opRestriction(I, ncol)
    Z = opRestriction(:, ncol)

Creates a LinearOperator restricting a `ncol`-sized vector to indices `I` with storage Vectors `Mv`, `Mtu`.
The operation `Z * v` is equivalent to `v[I]`. `I` can be `:`.

    Z = opRestriction(k, ncol)

Alias for `opRestriction([k], ncol)`.
"""
function opRestriction(Idx::LinearOperatorIndexType{I}, ncol::I) where {T,I<:Integer}
  all(1 .≤ Idx .≤ ncol) || throw(LinearOperatorException("indices should be between 1 and $ncol"))
  nrow = length(Idx)
  prod! = @closure (res, v, α, β) -> mulRestrict!(res, Idx, v, α, β)
  tprod! = @closure (res, u, α, β) -> multRestrict!(res, Idx, u, α, β)
  return LinearOperator{I}(nrow, ncol, false, false, prod!, tprod!, tprod!)
end

opRestriction(::Colon, ncol::I) where {I<:Integer} = opEye(I, ncol)

opRestriction(k::I, ncol::I) where {I<:Integer} = opRestriction([k], ncol)

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
opExtension(Idx::LinearOperatorIndexType{I}, ncol::I) where {I<:Integer} = opRestriction(Idx, ncol)'

opExtension(::Colon, ncol::I) where {I<:Integer} = opEye(I, ncol)

opExtension(k::I, ncol::I) where {I<:Integer} = opExtension([k], ncol)

# indexing for linear operators
import Base.getindex
function getindex(
  op::AbstractLinearOperator,
  rows::Union{LinearOperatorIndexType{I}, I, Colon},
  cols::Union{LinearOperatorIndexType{I}, I, Colon},
) where {I<:Integer}
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
function BlockDiagonalOperator(ops...) where {S}
  nrow = ncol = 0
  for op ∈ ops
    m, n = size(op)
    nrow += m
    ncol += n
  end
  T = promote_eltypeof(ops...)

  function prod!(y, x, α, β)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      @views mul!(y[(k + 1):(k + m)] , op, x[(j + 1):(j + n)], α, β)
      k += m
      j += n
    end
  end

  function tprod!(y, x, α, β)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      @views mul!(y[(k + 1):(k + n)], transpose(op), x[(j + 1):(j + m)], α, β)
      k += n
      j += m
    end
  end

  function ctprod!(y, x, α, β)
    k = 0
    j = 0
    for op ∈ ops
      m, n = size(op)
      @views mul!(y[(k + 1):(k + n)], adjoint(op), x[(j + 1):(j + m)], α, β)
      k += n
      j += m
    end
  end

  symm = all((issymmetric(op) for op ∈ ops))
  herm = all((ishermitian(op) for op ∈ ops))
  LinearOperator{T}(nrow, ncol, symm, herm, prod!, tprod!, ctprod!)
end
