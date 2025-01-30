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
*(
  v::Union{LinearAlgebra.Adjoint{S, V}, LinearAlgebra.Transpose{S, V}},
  ::LinearOperators.opEye,
) where {S, V <: AbstractVector{S}} = v
*(::opEye, T::AbstractLinearOperator) = T
*(T::AbstractLinearOperator, ::opEye) = T
*(::opEye, T::opEye) = T

adjoint(A::opEye) = A
transpose(A::opEye) = A
conj(A::opEye) = A

function show(io::IO, op::opEye)
  println(io, "Identity operator")
end

function mulOpEye!(res, v, α, β::T, n_min) where {T}
  if β == zero(T)
    res[1:n_min] .= @views α .* v[1:n_min]
    res[(n_min + 1):end] .= 0
  else
    res[1:n_min] .= @views α .* v[1:n_min] .+ β .* res[1:n_min]
    res[(n_min + 1):end] .= β
  end
end

"""
    opEye(T, n; S = Vector{T})
    opEye(n)

Identity operator of order `n` and of data type `T` (defaults to `Float64`).
Change `S` to use LinearOperators on GPU.
"""
function opEye(T::Type, n::Int; S = Vector{T})
  prod! = @closure (res, v, α, β) -> mulOpEye!(res, v, α, β, n)
  LinearOperator{T}(n, n, true, true, prod!, prod!, prod!, S = S)
end

opEye(n::Int) = opEye(Float64, n)

# TODO: not type stable
"""
    opEye(T, nrow, ncol; S = Vector{T})
    opEye(nrow, ncol)

Rectangular identity operator of size `nrow`x`ncol` and of data type `T`
(defaults to `Float64`).
Change `S` to use LinearOperators on GPU.
"""
function opEye(T::Type, nrow::I, ncol::I; S = Vector{T}) where {I <: Integer}
  if nrow == ncol
    return opEye(T, nrow; S = S)
  end
  prod! = @closure (res, v, α, β) -> mulOpEye!(res, v, α, β, min(nrow, ncol))
  return LinearOperator{T}(nrow, ncol, false, false, prod!, prod!, prod!, S = S)
end

opEye(nrow::I, ncol::I) where {I <: Integer} = opEye(Float64, nrow, ncol)

function mulOpOnes!(res, v, α, β::T) where {T}
  if β == zero(T)
    res .= (α * sum(v))
  else
    res .= (α * sum(v)) .+ β .* res
  end
end

"""
    opOnes(T, nrow, ncol; S = Vector{T})
    opOnes(nrow, ncol)

Operator of all ones of size `nrow`-by-`ncol` of data type `T` (defaults to
`Float64`).
Change `S` to use LinearOperators on GPU.
"""
function opOnes(T::Type, nrow::I, ncol::I; S = Vector{T}) where {I <: Integer}
  prod! = @closure (res, v, α, β) -> mulOpOnes!(res, v, α, β)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod!, prod!, prod!, S = S)
end

opOnes(nrow::I, ncol::I) where {I <: Integer} = opOnes(Float64, nrow, ncol)

function mulOpZeros!(res, v, α, β::T) where {T}
  if β == zero(T)
    res .= 0
  else
    res .*= β
  end
end

"""
    opZeros(T, nrow, ncol; S = Vector{T})
    opZeros(nrow, ncol)

Zero operator of size `nrow`-by-`ncol`, of data type `T` (defaults to
`Float64`).
Change `S` to use LinearOperators on GPU.
"""
function opZeros(T::Type, nrow::I, ncol::I; S = Vector{T}) where {I <: Integer}
  prod! = @closure (res, v, α, β) -> mulOpZeros!(res, v, α, β)
  LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol, prod!, prod!, prod!, S = S)
end

opZeros(nrow::I, ncol::I) where {I <: Integer} = opZeros(Float64, nrow, ncol)

function mulSquareOpDiagonal!(res, d, v, α, β::T) where {T}
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
  LinearOperator{T}(length(d), length(d), true, isreal(d), prod!, prod!, ctprod!, S = typeof(d))
end

function mulOpDiagonal!(res, d, v, α, β::T, n_min) where {T}
  if β == zero(T)
    res[1:n_min] .= @views α .* d[1:n_min] .* v[1:n_min]
  else
    res[1:n_min] .= @views α .* d[1:n_min] .* v[1:n_min] .+ β .* res[1:n_min]
  end
  res[(n_min + 1):end] .= 0
end
"""
    opDiagonal(nrow, ncol, d)

Rectangular diagonal operator of size `nrow`-by-`ncol` with the vector `d` on
its main diagonal.
"""
function opDiagonal(nrow::I, ncol::I, d::AbstractVector{T}) where {T, I <: Integer}
  nrow == ncol <= length(d) && (return opDiagonal(d[1:nrow]))
  n_min = min(nrow, ncol)
  prod! = @closure (res, v, α, β) -> mulOpDiagonal!(res, d, v, α, β, n_min)
  tprod! = @closure (res, u, α, β) -> mulOpDiagonal!(res, d, u, α, β, n_min)
  ctprod! = @closure (res, w, α, β) -> mulOpDiagonal!(res, conj.(d), w, α, β, n_min)
  LinearOperator{T}(nrow, ncol, false, false, prod!, tprod!, ctprod!, S = typeof(d))
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

Creates a LinearOperator restricting a `ncol`-sized vector to indices `I`.
The operation `Z * v` is equivalent to `v[I]`. `I` can be `:`.

    Z = opRestriction(k, ncol)

Alias for `opRestriction([k], ncol)`.
"""
function opRestriction(Idx::LinearOperatorIndexType{I}, ncol::I; S = nothing) where {I <: Integer}
  all(1 .≤ Idx .≤ ncol) || throw(LinearOperatorException("indices should be between 1 and $ncol"))
  nrow = length(Idx)
  prod! = @closure (res, v, α, β) -> mulRestrict!(res, Idx, v, α, β)
  tprod! = @closure (res, u, α, β) -> multRestrict!(res, Idx, u, α, β)
  if isnothing(S)
    return LinearOperator{I}(nrow, ncol, false, false, prod!, tprod!, tprod!)
  else
    return LinearOperator{I}(nrow, ncol, false, false, prod!, tprod!, tprod!; S = S)
  end
end

opRestriction(::Colon, ncol::I) where {I <: Integer} = opEye(I, ncol)

opRestriction(k::I, ncol::I) where {I <: Integer} = opRestriction([k], ncol)

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
opExtension(Idx::LinearOperatorIndexType{I}, ncol::I; S = nothing) where {I <: Integer} =
  opRestriction(Idx, ncol; S = S)'

opExtension(::Colon, ncol::I) where {I <: Integer} = opEye(I, ncol)

opExtension(k::I, ncol::I) where {I <: Integer} = opExtension([k], ncol)

# indexing for linear operators
import Base.getindex
function getindex(
  op::AbstractLinearOperator,
  rows::Union{LinearOperatorIndexType{<:Integer}, <:Integer, Colon},
  cols::Union{LinearOperatorIndexType{<:Integer}, <:Integer, Colon},
)
  R = opRestriction(rows, size(op, 1))
  E = opExtension(cols, size(op, 2))
  return R * op * E
end

eltypeof(op::AbstractLinearOperator) = eltype(op)  # need this for promote_eltypeof

"""
    BlockDiagonalOperator(M1, M2, ..., Mn; S = promote_type(storage_type.(M1, M2, ..., Mn)))

Creates a block-diagonal linear operator:

    [ M1           ]
    [    M2        ]
    [       ...    ]
    [           Mn ]

Change `S` to use LinearOperators on GPU.
"""
function BlockDiagonalOperator(ops...; S = promote_type(storage_type.(ops)...))
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
      @views mul!(y[(k + 1):(k + m)], op, x[(j + 1):(j + n)], α, β)
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
  args5 = all((has_args5(op) for op ∈ ops))
  CompositeLinearOperator(T, nrow, ncol, symm, herm, prod!, tprod!, ctprod!, args5, S = S)
end
