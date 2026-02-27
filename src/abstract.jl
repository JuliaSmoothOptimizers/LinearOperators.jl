export AbstractLinearOperator,
  AbstractQuasiNewtonOperator,
  AbstractDiagonalQuasiNewtonOperator,
  LinearOperator,
  LinearOperatorException,
  hermitian,
  ishermitian,
  symmetric,
  issymmetric,
  has_args5,     # TODO: deprecate
  isallocated5,
  nprod,
  ntprod,
  nctprod,
  reset!

struct LinearOperatorException <: Exception
  msg::AbstractString
end

# when indexing, Colon() is treated separately
const LinearOperatorIndexType{I} =
  Union{UnitRange{I}, StepRange{I, I}, AbstractVector{I}} where {I <: Integer}

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import Base.copy!, Base.copyto!
import Base.Broadcast: BroadcastStyle, DefaultArrayStyle, Broadcasted, broadcastable, materialize!
import LinearAlgebra.Symmetric,
  LinearAlgebra.issymmetric, LinearAlgebra.Hermitian, LinearAlgebra.ishermitian

abstract type AbstractLinearOperator{T} end
abstract type AbstractQuasiNewtonOperator{T} <: AbstractLinearOperator{T} end
abstract type AbstractDiagonalQuasiNewtonOperator{T} <: AbstractQuasiNewtonOperator{T} end
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
mutable struct LinearOperator{T, S, I <: Integer, F, Ft, Fct} <: AbstractLinearOperator{T}
  const nrow::I
  const ncol::I
  const symmetric::Bool
  const hermitian::Bool
  const prod!::F
  const tprod!::Ft
  const ctprod!::Fct
  nprod::I
  ntprod::I
  nctprod::I
  Mv::S
  Mtu::S
end

function LinearOperator{T, S}(
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
) where {T, S, I <: Integer, F, Ft, Fct}
  return LinearOperator{T, S, I, F, Ft, Fct}(
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
    S(undef, 0),
    S(undef, 0),
  )
end

# backward compatibility (not inferrable; use LinearOperator{T, S} if you want something inferrable)
LinearOperator{T}(
  nrow,
  ncol,
  symmetric,
  hermitian,
  prod!,
  tprod!,
  ctprod!,
  nprod,
  ntprod,
  nctprod;
  S::Type = Vector{T},
) where {T} = LinearOperator{T, S}(
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
)

"""
    LinearOperator{T, S}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod! = nothing) where {T, S}

Construct a linear operator with a specific temporary array storage type `S`, which should typically have element type `T`.

This is an inferrable variant of constructors that supply `S` as a keyword argument, and is recommended for performance-sensitive applications.
"""
LinearOperator{T, S}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod! = nothing,
  ctprod! = nothing,
) where {T, S, I <: Integer} =
  LinearOperator{T, S}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, 0, 0, 0)

# backward compatibility (not inferrable)
LinearOperator{T}(
  nrow,
  ncol,
  symmetric,
  hermitian,
  prod!,
  tprod!,
  ctprod!;
  S::Type = Vector{T},
) where {T} = LinearOperator{T, S}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!)

const CompositeLinearOperator = LinearOperator   # backwards compatibility

nprod(op::AbstractLinearOperator) = op.nprod
ntprod(op::AbstractLinearOperator) = op.ntprod
nctprod(op::AbstractLinearOperator) = op.nctprod

increase_nprod!(op::AbstractLinearOperator) = (op.nprod += 1)
increase_ntprod!(op::AbstractLinearOperator) = (op.ntprod += 1)
increase_nctprod!(op::AbstractLinearOperator) = (op.nctprod += 1)

"""
    has_args5(op)

Determine whether the operator can work with the 5-args `mul!`.
If `false`, storage vectors will be generated at the first call of
the 5-args `mul!`.
No additional vectors are generated when using the 3-args `mul!`.

!!! warning
    `has_args5` can be very slow. A better option is to use Julia's `hasmethod`
    at points in the code where the concrete types of objects used in `mul!` are known.

    `has_args5` may be removed in a future release.
"""
has_args5(op::AbstractLinearOperator) = get_nargs(op.prod!) == 4

isallocated5(op::LinearOperator) = !(isempty(op.Mv) || isempty(op.Mtu))

has_args5(op::AbstractMatrix) = true  # Needed for BlockDiagonalOperator

# Alert user of the need for storage_type method definition for arbitrary, user defined operators
storage_type(op::AbstractLinearOperator) = error("please implement storage_type for $(typeof(op))")

storage_type(::LinearOperator{T, S}) where {T, S} = S
storage_type(::AbstractMatrix{T}) where {T} = Vector{T}

# Lazy wrappers
storage_type(op::Adjoint) = storage_type(parent(op))
storage_type(op::Transpose) = storage_type(parent(op))
storage_type(op::Diagonal) = typeof(parent(op))

broadcastable(op::AbstractLinearOperator) = Ref(op)
BroadcastStyle(::Type{<:AbstractLinearOperator}) = DefaultArrayStyle{0}()

function copyto!(
  dest::LinearOperator{T, S},
  src::LinearOperator{T, S},
) where {T, S}
  dest.nrow = src.nrow
  dest.ncol = src.ncol
  dest.symmetric = src.symmetric
  dest.hermitian = src.hermitian
  dest.prod! = src.prod!
  dest.tprod! = src.tprod!
  dest.ctprod! = src.ctprod!
  dest.nprod = src.nprod
  dest.ntprod = src.ntprod
  dest.nctprod = src.nctprod
  dest.args5 = src.args5
  dest.use_prod5! = src.use_prod5!
  dest.Mv5 = src.Mv5
  dest.Mtu5 = src.Mtu5
  dest.allocated5 = src.allocated5
  return dest
end

function copyto!(dest::LinearOperator, src::LinearOperator)
  throw(
    LinearOperatorException(
      "cannot update a LinearOperator in-place: incompatible element types " *
      "$(eltype(dest)) and $(eltype(src)), or storage types " *
      "$(typeof(dest.Mv5)) and $(typeof(src.Mv5)). " *
      "Use assignment (`dest = src`) instead.",
    ),
  )
end

copy!(dest::LinearOperator, src::LinearOperator) = copyto!(dest, src)

function materialize!(dest::LinearOperator, bc::Broadcasted{DefaultArrayStyle{0}})
  (bc.f === identity && length(bc.args) == 1) ||
    throw(
      LinearOperatorException(
        "only broadcast assignment of a single LinearOperator is supported (e.g., `op .= new_op`).",
      ),
    )
  src_arg = bc.args[1]
  src = src_arg isa Ref ? src_arg[] : src_arg
  src isa LinearOperator ||
    throw(LinearOperatorException("right-hand side of `op .= ...` must be a LinearOperator"))
  return copyto!(dest, src)
end

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
