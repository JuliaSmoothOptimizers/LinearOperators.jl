export AbstractLinearOperator,
  LinearOperator,
  LinearOperatorException,
  hermitian,
  ishermitian,
  symmetric,
  issymmetric,
  has_args5,
  isallocated5,
  nprod,
  ntprod,
  nctprod,
  reset!,
  shape

mutable struct LinearOperatorException <: Exception
  msg::AbstractString
end

# when indexing, Colon() is treated separately
const LinearOperatorIndexType{I} =
  Union{UnitRange{I}, StepRange{I, I}, AbstractVector{I}} where {I <: Integer}

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import LinearAlgebra.Symmetric,
  LinearAlgebra.issymmetric, LinearAlgebra.Hermitian, LinearAlgebra.ishermitian

abstract type AbstractLinearOperator{T} end
OperatorOrMatrix = Union{AbstractLinearOperator, AbstractMatrix}

eltype(A::AbstractLinearOperator{T}) where {T} = T
isreal(A::AbstractLinearOperator{T}) where {T} = T <: Real
get_nargs(f) = first(methods(f)).nargs - 1

"""
Base type to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
"""
mutable struct LinearOperator{T, I <: Integer, F, Ft, Fct, S} <: AbstractLinearOperator{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::Ft
  ctprod!::Fct
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::S
  Mtu5::S
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
end

function LinearOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  nprod::I,
  ntprod::I,
  nctprod::I,
) where {T, I <: Integer, F, Ft, Fct}
  Mv5, Mtu5 = T[], T[]
  S = typeof(Mv5)
  nargs = get_nargs(prod!)
  args5 = (nargs == 4)
  (args5 == false) || (nargs != 2) || throw(LinearOperatorException("Invalid number of arguments"))
  allocated5 = args5 ? true : false
  use_prod5! = args5 ? true : false
  return LinearOperator{T, I, F, Ft, Fct, S}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    nprod,
    ntprod,
    nctprod,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5,
  )
end

LinearOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod!,
  ctprod!,
) where {T, I <: Integer} =
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, 0, 0, 0)

# create operator from other operators with +, *, vcat,...
function CompositeLinearOperator(
  T::DataType,
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  args5::Bool,
) where {I <: Integer, F, Ft, Fct}
  Mv5, Mtu5 = T[], T[]
  S = typeof(Mv5)
  allocated5 = true
  use_prod5! = true
  return LinearOperator{T, I, F, Ft, Fct, S}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    0,
    0,
    0,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5,
  )
end

nprod(op::AbstractLinearOperator) = op.nprod
ntprod(op::AbstractLinearOperator) = op.ntprod
nctprod(op::AbstractLinearOperator) = op.nctprod

increase_nprod(op::AbstractLinearOperator) = (op.nprod += 1)
increase_ntprod(op::AbstractLinearOperator) = (op.ntprod += 1)
increase_nctprod(op::AbstractLinearOperator) = (op.nctprod += 1)

"""
    has_args5(op)

Determine whether the operator can work with the 5-args `mul!`.
If `false`, storage vectors will be generated at the first call of
the 5-args `mul!`.
No additional vectors are generated when using the 3-args `mul!`.
"""
has_args5(op::LinearOperator) = op.args5
use_prod5!(op::LinearOperator) = op.use_prod5!
isallocated5(op::LinearOperator) = op.allocated5

has_args5(op::AbstractMatrix) = true  # Needed for BlockDiagonalOperator

"""
  reset!(op)

Reset the product counters of a linear operator.
"""
function reset!(op::AbstractLinearOperator)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end

"""
    m, n = size(op)

Return the size of a linear operator as a tuple.
"""
size(op::AbstractLinearOperator) = (op.nrow, op.ncol)

"""
    m = size(op, d)

Return the size of a linear operator along dimension `d`.
"""
function size(op::AbstractLinearOperator, d::Integer)
  nrow, ncol = size(op)
  if d == 1
    return nrow
  end
  if d == 2
    return ncol
  end
  throw(LinearOperatorException("Linear operators only have 2 dimensions for now"))
end

"""
    m, n = shape(op)

An alias for size.
"""
shape(op::AbstractLinearOperator) = size(op)

"""
    ishermitian(op)

Determine whether the operator is Hermitian.
"""
ishermitian(op::AbstractLinearOperator) = op.hermitian

"""
    Hermitian(op, uplo=:U)
"""
function Hermitian(op::AbstractLinearOperator, uplo::Symbol = :U)
  isequal(size(op)...) || throw(LinearOperatorException("Operator is not square"))
  ishermitian(op) && return op
  return (op + adjoint(op)) / 2
end

hermitian(A::AbstractLinearOperator, uplo::Symbol) = Hermitian(A, uplo)

"""
    issymmetric(op)

Determine whether the operator is symmetric.
"""
issymmetric(op::AbstractLinearOperator) = op.symmetric

"""
    Symmetric(op, uplo=:U)
"""
function Symmetric(op::AbstractLinearOperator, uplo::Symbol = :U)
  isequal(size(op)...) || throw(LinearOperatorException("Operator is not square"))
  issymmetric(op) && return op
  return (op + transpose(op)) / 2
end

symmetric(A::AbstractLinearOperator, uplo::Symbol) = Symmetric(A, uplo)

"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io::IO, op::AbstractLinearOperator)
  s = "Linear operator\n"
  nrow, ncol = size(op)
  s *= @sprintf("  nrow: %s\n", nrow)
  s *= @sprintf("  ncol: %d\n", ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", issymmetric(op))
  s *= @sprintf("  hermitian: %s\n", ishermitian(op))
  s *= @sprintf("  nprod:   %d\n", nprod(op))
  s *= @sprintf("  ntprod:  %d\n", ntprod(op))
  s *= @sprintf("  nctprod: %d\n", nctprod(op))
  s *= "\n"
  print(io, s)
end

"""
    A = Matrix(op)

Materialize an operator as a dense array using `op.ncol` products.
"""
function Base.Matrix(op::AbstractLinearOperator{T}) where {T}
  (m, n) = size(op)
  A = Array{T}(undef, m, n)
  ei = zeros(T, n)
  for i = 1:n
    ei[i] = one(T)
    A[:, i] = op * ei
    ei[i] = zero(T)
  end
  return A
end
