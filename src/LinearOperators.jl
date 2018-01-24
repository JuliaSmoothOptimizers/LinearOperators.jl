__precompile__()
# Linear Operators for Julia
module LinearOperators

using Compat

export AbstractLinearOperator, LinearOperator,
       LinearOperatorException,
       A_mul_B!, At_mul_B!, Ac_mul_B!,
       opEye, opOnes, opZeros, opDiagonal,
       opInverse, opCholesky, opLDL, opHouseholder, opHermitian,
       check_ctranspose, check_hermitian, check_positive_definite,
       shape, hermitian, ishermitian, symmetric, issymmetric,
       opRestriction, opExtension


mutable struct LinearOperatorException <: Exception
  msg :: AbstractString
end

# when indexing, Colon() is treated separately
const LinearOperatorIndexType = Union{UnitRange{Int}, StepRange{Int, Int}, AbstractVector{Int}}

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import Base.+, Base.-, Base.*
import Base.A_mul_B!, Base.At_mul_B!, Base.Ac_mul_B!
import Base.transpose, Base.ctranspose
import Base.full
import Base.conj
import Base.issymmetric, Base.ishermitian
import Base.hcat, Base.vcat

@compat abstract type AbstractLinearOperator{T} end
OperatorOrMatrix = Union{AbstractLinearOperator, AbstractMatrix}

eltype{T}(A :: AbstractLinearOperator{T}) = T
isreal{T}(A :: AbstractLinearOperator{T}) = T <: Real


"""
Base type to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
"""
mutable struct LinearOperator{T} <: AbstractLinearOperator{T}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
end

"""
    m, n = size(op)

Return the size of a linear operator as a tuple.
"""
size(op :: AbstractLinearOperator) = (op.nrow, op.ncol)

"""
    m = size(op, d)

Return the size of a linear operator along dimension `d`.
"""
function size(op :: AbstractLinearOperator, d :: Int)
  if d == 1
    return op.nrow
  end
  if d == 2
    return op.ncol
  end
  throw(LinearOperatorException("Linear operators only have 2 dimensions for now"))
end

"""
    m, n = shape(op)

An alias for size.
"""
shape(op :: AbstractLinearOperator) = size(op)

"""
    hermitian(op)
    ishermitian(op)

Determine whether the operator is Hermitian.
"""
hermitian(op :: AbstractLinearOperator) = op.hermitian
ishermitian(op :: AbstractLinearOperator) = op.hermitian

"""
    symmetric(op)
    issymmetric(op)

Determine whether the operator is symmetric.
"""
symmetric(op :: AbstractLinearOperator) = op.symmetric
issymmetric(op :: AbstractLinearOperator) = op.symmetric


"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io :: IO, op :: AbstractLinearOperator)
  s  = "Linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  prod:   %s\n", string(op.prod))
  s *= @sprintf("  tprod:  %s\n", string(op.tprod))
  s *= @sprintf("  ctprod: %s", string(op.ctprod))
  s *= "\n"
  print(io, s)
end


# Constructors.
"""
    LinearOperator(M; symmetric=false, hermitian=false)

Construct a linear operator from a dense or sparse matrix.
Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
LinearOperator{T}(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) =
  LinearOperator{T}(size(M)..., symmetric, hermitian,
                    v -> M * v,
                    Nullable{Function}(u -> M.' * u),
                    Nullable{Function}(w -> M' * w))

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
LinearOperator{T}(M :: SymTridiagonal{T}) =
    LinearOperator(M; symmetric=true, hermitian=eltype(M) <: Real)

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
LinearOperator{T}(M :: Symmetric{T}) =
    LinearOperator(M; symmetric=true, hermitian=eltype(M) <: Real)

"""
    LinearOperator(M)

Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
LinearOperator{T}(M :: Hermitian{T}) =
    LinearOperator(M; symmetric=eltype(M) <: Real, hermitian=true)

# the only advantage of this constructor is optional args
# use LinearOperator{Float64} if you mean real instead of complex
"""
    LinearOperator(nrow, ncol, symmetric, hermitian, prod,
                    [tprod=Nullable{Function}(),
                    ctprod=Nullable{Function}()])

Construct a linear operator from functions.
"""
function LinearOperator(nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod :: Function,
                        tprod :: Union{Function,Nullable{Function}}=Nullable{Function}(),
                        ctprod :: Union{Function,Nullable{Function}}=Nullable{Function}())

  T = hermitian ? (symmetric ? Float64 : Complex128) : Complex128
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end


# Apply an operator to a vector.
function *(op :: AbstractLinearOperator, v :: AbstractVector)
  size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
  op.prod(v)
end


"""
    A = full(op)

Materialize an operator as a dense array using `op.ncol` products.
"""
function full(op :: AbstractLinearOperator)
  (m, n) = size(op)
  A = Array{eltype(op)}(m, n)
  ei = zeros(eltype(op), n)
  for i = 1 : n
    ei[i] = 1
    A[:, i] = op * ei
    ei[i] = 0
  end
  return A
end

# Unary operations.
+(op :: AbstractLinearOperator) = op
-{T}(op :: AbstractLinearOperator{T}) = LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian,
                                                          v -> -op.prod(v),
                                                          Nullable{Function}(u -> -get(op.tprod)(u)),
                                                          Nullable{Function}(w -> -get(op.ctprod)(w)))

function transpose{T}(op :: AbstractLinearOperator{T})
  if op.symmetric
    return op
  end
  if !isnull(op.tprod)
    return LinearOperator{T}(op.ncol, op.nrow, op.symmetric, op.hermitian,
                             get(op.tprod),
                             Nullable{Function}(op.prod),
                             Nullable{Function}(v -> conj(get(op.tprod)(v))))
  end
  if isnull(op.ctprod)
    if op.hermitian
      ctprod = op.prod
    else
      throw(LinearOperatorException("unable to infer transpose operator"))
    end
  else
    ctprod = get(op.ctprod)
  end

  return LinearOperator{T}(op.ncol, op.nrow, op.symmetric, op.hermitian,
                           v -> conj(ctprod(conj(v))),                # A.'v = conj(A' conj(v))
                           Nullable{Function}(op.prod),               # (A.').' = A
                           Nullable{Function}(w -> conj(op.prod(w)))) # (A.')' = conj(A)
end

function ctranspose{T}(op :: LinearOperator{T})
  if op.hermitian
    return op
  end
  if !isnull(op.ctprod)
    return LinearOperator{T}(op.ncol, op.nrow, op.symmetric, op.hermitian,
                             get(op.ctprod),
                             Nullable{Function}(u -> conj(op.prod(u))),
                             Nullable{Function}(op.prod))
  end
  if isnull(op.tprod)
    if op.symmetric
      tprod = op.prod
    else
      throw(LinearOperatorException("unable to infer conjugate transpose operator"))
    end
  else
    tprod = get(op.tprod)
  end

  return LinearOperator{T}(op.ncol, op.nrow, op.symmetric, op.hermitian,
                           v -> conj(tprod(v)),
                           Nullable{Function}(u -> conj(op.prod(u))),
                           Nullable{Function}(op.prod))
end

function conj{T}(op :: AbstractLinearOperator{T})
  return LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian,
                           v -> conj(op.prod(conj(v))),
                           op.ctprod,
                           op.tprod)
end

function A_mul_B!(y :: AbstractVector, op :: AbstractLinearOperator, x :: AbstractVector)
  y[:] = op * x
  return y
end

function At_mul_B!(y :: AbstractVector, op :: AbstractLinearOperator, x :: AbstractVector)
  y[:] = op.' * x
  return y
end

function Ac_mul_B!(y :: AbstractVector, op :: AbstractLinearOperator, x :: AbstractVector)
  y[:] = op' * x
  return y
end

# Binary operations.

## Operator times operator.
function *(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  return LinearOperator{S}(m1, n2, false, false,
                           v -> op1 * (op2 * v),
                           u -> op2.' * (op1.' * u),
                           w -> op2' * (op1' * w))
end

## Matrix times operator.
*(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) * op
*(op :: AbstractLinearOperator, M :: AbstractMatrix) = op * LinearOperator(M)

## Scalar times operator.
function *(op :: AbstractLinearOperator, x :: Number)
  S = promote_type(eltype(op), typeof(x))
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x),
                    v -> (op * v) * x,
                    u -> x * (op.' * u),
                    w -> x' * (op' * w))
end

function *(x :: Number, op :: AbstractLinearOperator)
  S = promote_type(eltype(op), typeof(x))
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x),
                    v -> x * (op * v),
                    u -> (op.' * u) * x,
                    w -> (op' * w) * x')
end

@static if VERSION < v"0.6.0-"
  .*(op :: AbstractLinearOperator, x :: Number) = op * x
  .*(x :: Number, op :: AbstractLinearOperator) = x * op
end

# Operator + operator.
function +(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  return LinearOperator{S}(m1, n1,
                           op1.symmetric && op2.symmetric,
                           op1.hermitian && op2.hermitian,
                           v -> (op1   * v) + (op2   * v),
                           u -> (op1.' * u) + (op2.' * u),
                           w -> (op1'  * w) + (op2'  * w))
end

# Operator + matrix.
+(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) + op
+(op :: AbstractLinearOperator, M :: AbstractMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
@static if VERSION < v"0.6.0-"
  .+(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
  .+(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op
else
  +(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
  +(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op
end

# Operator - operator
-(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) - op
-(op :: AbstractLinearOperator, M :: AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
@static if VERSION < v"0.6.0-"
  .-(op :: AbstractLinearOperator, x :: Number) = op .+ (-x)
  .-(x :: Number, op :: AbstractLinearOperator) = x .+ (-op)
else
  -(op :: AbstractLinearOperator, x :: Number) = op + (-x)
  -(x :: Number, op :: AbstractLinearOperator) = x + (-op)
end


# Utility functions.

"""
    check_ctranspose(op)

Cheap check that the operator and its conjugate transposed are related.
"""
function check_ctranspose(op :: AbstractLinearOperator)
  (m, n) = size(op)
  x = rand(n)
  y = rand(m)
  yAx = dot(y, op * x)
  xAty = dot(x, op' * y)
  ε = eps(Float64)
  return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1/3)
end

check_ctranspose(M :: AbstractMatrix) = check_ctranspose(LinearOperator(M))

"""
    check_hermitian(op)

Cheap check that the operator is Hermitian.
"""
function check_hermitian(op :: AbstractLinearOperator)
  m, n = size(op)
  v = rand(n)
  w = op * v
  s = dot(w, w);  # = (Av)'(Av) = v' A' A v.
  y = op * w
  t = dot(v, y);  # = v' A A v.
  ε = eps(Float64)
  return abs(s - t) < (abs(s) + ε) * ε^(1/3)
end

check_hermitian(M :: AbstractMatrix) = check_hermitian(LinearOperator(M))

"""
    check_positive_definite(op; semi=false)

Cheap check that the operator is positive (semi-)definite.
"""
function check_positive_definite(op :: AbstractLinearOperator; semi=false)
  m, n = size(op)
  v = rand(n)
  w = op * v
  vw = dot(v, w)
  ε = eps(Float64)
  if imag(vw) > sqrt(ε) * abs(vw)
    return false
  end
  vw = real(vw)
  return semi ? (vw ≥ 0) : (vw > 0)
end

check_positive_definite(M :: AbstractMatrix) = check_positive_definite(LinearOperator(M))

# Special linear operators.

"""
    opEye(T, n)
    opEye(n)

Identity operator of order `n` and of data type `T` (defaults to `Float64`).
"""
opEye(T :: DataType, n :: Int) = LinearOperator{T}(n, n, true, true,
                                                   v -> v[:], u -> u[:], w -> w[:])
opEye(n :: Int) = opEye(Float64, n)

"""
    opEye(T, nrow, ncol)
    opEye(nrow, ncol)

Rectangular identity operator of size `nrow`x`ncol` and of data type `T`
(defaults to `Float64`).
"""
function opEye(T :: DataType, nrow :: Int, ncol :: Int)
  nrow == ncol && opEye(T, nrow)
  if nrow > ncol
    return LinearOperator{T}(nrow, ncol, false, false,
                             v -> [v ; zeros(nrow - ncol)],
                             v -> v[1:ncol],
                             v -> v[1:ncol])
  else
    return LinearOperator{T}(nrow, ncol, false, false,
                             v -> v[1:nrow],
                             v -> [v ; zeros(ncol - nrow)],
                             v -> [v ; zeros(ncol - nrow)])
  end
end

opEye(nrow :: Int, ncol :: Int) = opEye(Float64, nrow, ncol)

"""
    opOnes(T, nrow, ncol)
    opOnes(nrow, ncol)

Operator of all ones of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
opOnes(T :: DataType, nrow :: Int, ncol :: Int) = LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol,
                                                                    v -> sum(v) * ones(nrow),
                                                                    u -> sum(u) * ones(ncol),
                                                                    w -> sum(w) * ones(ncol))
opOnes(nrow :: Int, ncol :: Int) = opOnes(Float64, nrow, ncol)

"""
    opZeros(T, nrow, ncol)
    opZeros(nrow, ncol)

Zero operator of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
opZeros(T :: DataType, nrow :: Int, ncol :: Int) = LinearOperator{T}(nrow, ncol,
                                                                     nrow == ncol, nrow == ncol,
                                                                     v -> zeros(nrow),
                                                                     u -> zeros(ncol),
                                                                     w -> zeros(ncol))
opZeros(nrow :: Int, ncol :: Int) = opZeros(Float64, nrow, ncol)

"""
    opDiagonal(d)

Diagonal operator with the vector `d` on its main diagonal.
"""
opDiagonal{T}(d :: AbstractVector{T}) = LinearOperator{T}(length(d), length(d), true, isreal(d),
                                                          v -> v .* d,
                                                          u -> u .* d,
                                                          w -> w .* conj(d))

"""
    opDiagonal(nrow, ncol, d)

Rectangular diagonal operator of size `nrow`-by-`ncol` with the vector `d` on
its main diagonal.
"""
function opDiagonal{T}(nrow :: Int, ncol :: Int, d :: AbstractVector{T})
  nrow == ncol <= length(d) && (return opDiagonal(d[1:nrow]))
  if nrow > ncol
    D = LinearOperator{T}(nrow, ncol, false, false,
                          v -> [v .* d ; zeros(nrow-ncol)],
                          u -> u[1:ncol] .* d,
                          w -> w[1:ncol] .* conj(d))
  else
    D = LinearOperator{T}(nrow, ncol, false, false,
                          v -> v[1:nrow] .* d,
                          u -> [u .* d ; zeros(ncol-nrow)],
                          w -> [w .* conj(d) ; zeros(ncol-nrow)])
  end
  return D
end


hcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = hcat(A, LinearOperator(B))

hcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = hcat(LinearOperator(A), B)

function hcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.nrow == B.nrow || throw(LinearOperatorException("hcat: inconsistent row sizes"))

  nrow  = A.nrow
  ncol  = A.ncol + B.ncol
  S = promote_type(eltype(A), eltype(B))

  prod(v)   =  A * v[1:A.ncol] + B * v[A.ncol+1:end]
  tprod(v)  =  [A.' * v; B.' * v;]
  ctprod(v) =  [A' * v; B' * v;]

  LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function hcat(ops :: OperatorOrMatrix...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end


vcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = vcat(A, LinearOperator(B))

vcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = vcat(LinearOperator(A), B)

function vcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.ncol == B.ncol || throw(LinearOperatorException("vcat: inconsistent column sizes"))

  nrow  = A.nrow + B.nrow
  ncol  = A.ncol
  S = promote_type(eltype(A), eltype(B))

  prod(v)   =  [A * v; B * v;]
  tprod(v)  =  A.' * v +  B.' * v
  ctprod(v) =  A' * v[1:A.nrow] + B' * v[A.nrow+1:end]

  return LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function vcat(ops :: OperatorOrMatrix...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end


"""
    opInverse(M; symmetric=false, hermitian=false)

Inverse of a matrix as a linear operator using `\\`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`.
"""
opInverse{T}(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) =
  LinearOperator{T}(size(M,2), size(M,1), symmetric, hermitian,
                    v -> M \ v, u -> M.' \ u, w -> M' \ w)

"""
    opCholesky(M, [check=false])

Inverse of a Hermitian and positive definite matrix as a linear operator
using its Cholesky factorization. The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks.
"""
function opCholesky(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
    check_positive_definite(M) || throw(LinearOperatorException("matrix is not positive definite"))
  end
  if issparse(M)
    LL = cholfact(M)
    return LinearOperator{eltype(LL)}(m, m, isreal(M), true,
                                      v -> LL \ v,
                                      u -> conj(LL \ conj(u)),  # M.' = conj(M)
                                      w -> LL \ w)
  else
    L = chol(M)'
    return LinearOperator{eltype(L)}(m, m, isreal(M), true,
                                     v -> L' \ (L \ v),
                                     u -> L.' \ (conj(L \ conj(u))),
                                     w -> L' \ (L \ w))
  end
  #TODO: use iterative refinement.
end

"""
    opLDL(M, [check=false])

Inverse of a symmetric matrix as a linear operator using its LDL' factorization
if it exists. The factorization is computed only once. The optional `check`
argument will perform a cheap hermicity check.
"""
function opLDL(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldltfact(M)
  return LinearOperator{eltype(LDL)}(m, m, isreal(M), true,
                                     v -> LDL \ v,
                                     u -> conj(LDL \ conj(u)),  # M.' = conj(M)
                                     w -> LDL \ w)
  #TODO: use iterative refinement.
end

"""
    opHouseholder(h)

Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h h') x`.
"""
opHouseholder{T}(h :: AbstractVector{T}) = LinearOperator{T}(length(h), length(h), isreal(h), true,
                                                             v -> (v - 2 * dot(h, v) * h),
                                                             Nullable{Function}(),  # Will be inferred.
                                                             w -> (w - 2 * dot(h, w) * h))


"""
    opHermitian(d, A)

A symmetric/hermitian operator based on the diagonal `d` and lower triangle of `A`.
"""
function opHermitian{S,T}(d :: AbstractVector{S}, A :: AbstractMatrix{T})
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  return LinearOperator{U}(m, m, isreal(A), true,
                           v -> (d .* v + L * v + (v' * L)')[:],
                           Nullable{Function}(),
                           Nullable{Function}())
end


"""
    opHermitian(A)

A symmetric/hermitian operator based on a matrix.
"""
function opHermitian(T :: AbstractMatrix)
  d = diag(T)
  return opHermitian(d, T)
end

include("qn.jl")  # quasi-Newton operators
include("kron.jl")

"""
    Z = opRestriction(I, ncol)
    Z = opRestriction(:, ncol)

Creates a LinearOperator restricting a `ncol`-sized vector to indices `I`.
The operation `Z * v` is equivalent to `v[I]`. `I` can be `:`.

    Z = opRestriction(k, ncol)

Alias for `opRestriction([k], ncol)`.
"""
function opRestriction(I :: LinearOperatorIndexType, ncol :: Int)
  all(1 .≤ I .≤ ncol) || throw(LinearOperatorException("indices should be between 1 and $ncol"))
  nrow = length(I)
  tprod(x) = begin
    z = zeros(eltype(x), ncol)
    z[I] = x
    return z
  end
  return LinearOperator{Int}(nrow, ncol, false, false, x -> x[I], tprod, tprod)
end

opRestriction(::Colon, ncol :: Int) = opEye(Int, ncol)

opRestriction(k :: Int, ncol :: Int) = opRestriction([k], ncol)

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
opExtension(I :: LinearOperatorIndexType, ncol :: Int) = opRestriction(I, ncol)'

opExtension(::Colon, ncol :: Int) = opEye(Int, ncol)

opExtension(k :: Int, ncol :: Int) = opExtension([k], ncol)

# indexing for linear operators
import Base.getindex
function getindex(op :: AbstractLinearOperator,
                  rows :: Union{LinearOperatorIndexType, Int, Colon},
                  cols :: Union{LinearOperatorIndexType, Int, Colon})
  R = opRestriction(rows, size(op, 1))
  E = opExtension(cols, size(op, 2))
  return R * op * E
end

end  # module
