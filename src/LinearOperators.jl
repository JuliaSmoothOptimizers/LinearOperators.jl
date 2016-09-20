# Linear Operators for Julia
module LinearOperators

export AbstractLinearOperator, LinearOperator,
       LinearOperatorException,
       opEye, opOnes, opZeros, opDiagonal,
       opInverse, opCholesky, opLDL, opHouseholder, opHermitian,
       check_ctranspose, check_hermitian, check_positive_definite,
       shape, hermitian, symmetric,
       RestrictionOperator, ExtensionOperator


type LinearOperatorException <: Exception
  msg :: AbstractString
end

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import Base.+, Base.-, Base.*, Base.(.+), Base.(.-), Base.(.*)
import Base.transpose, Base.ctranspose
import Base.full
import Base.conj
import Base.hcat, Base.vcat

abstract AbstractLinearOperator{T}

eltype{T}(A :: AbstractLinearOperator{T}) = T
isreal{T}(A :: AbstractLinearOperator{T}) = T <: Real


"""
Base type to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
"""
type LinearOperator{T} <: AbstractLinearOperator{T}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
end


"Return the size of a linear operator as a tuple"
size(op :: AbstractLinearOperator) = (op.nrow, op.ncol)

"Return the size of a linear operator along dimension `d`"
function size(op :: AbstractLinearOperator, d :: Int)
  if d == 1
    return op.nrow
  end
  if d == 2
    return op.ncol
  end
  throw(LinearOperatorException("Linear operators only have 2 dimensions for now"))
end

"An alias for size"
shape(op :: AbstractLinearOperator) = size(op)

"Determine whether the operator is Hermitian"
hermitian(op :: AbstractLinearOperator) = op.hermitian

"Determine whether the operator is symmetric"
symmetric(op :: AbstractLinearOperator) = op.symmetric


"Display basic information about a linear operator"
function show(io :: IO, op :: AbstractLinearOperator)
  s  = "Linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  dtype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  prod:   %s\n", string(op.prod))
  s *= @sprintf("  tprod:  %s\n", string(op.tprod))
  s *= @sprintf("  ctprod: %s", string(op.ctprod))
  s *= "\n"
  print(io, s)
end


# Constructors.
"""Construct a linear operator from a dense or sparse matrix.
Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian."""
LinearOperator{T}(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) =
  LinearOperator{T}(size(M)..., symmetric, hermitian,
                    v -> M * v,
                    Nullable{Function}(u -> M.' * u),
                    Nullable{Function}(w -> M' * w))

"""Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric."""
LinearOperator{T}(M :: SymTridiagonal{T}) =
    LinearOperator(M; symmetric=true, hermitian=eltype(M) <: Real)

# the only advantage of this constructor is optional args
# use LinearOperator{Float64} if you mean real instead of complex
"Construct a linear operator from functions."
function LinearOperator(nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod :: Function,
                        tprod :: Union{Function,Nullable{Function}}=Nullable{Function}(),
                        ctprod :: Union{Function,Nullable{Function}}=Nullable{Function}())

  T = hermitian ? (symmetric ? Float64 : Complex128) : Complex128
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

# "Construct a real symmetric or complex hermitian linear operator from a function."
# LinearOperator(nrow :: Int, T :: DataType, prod :: Function) =
#   LinearOperator{T}(nrow, nrow, T <: Real, T <: Real || T <: Complex,
#                     prod, Nullable{Function}(prod), Nullable{Function}(prod))

# "Construct a linear operator from a single function."
# LinearOperator(nrow :: Int, ncol :: Int,
#                symmetric :: Bool, hermitian :: Bool,
#                prod :: Function) =
#   LinearOperator(nrow, ncol, symmetric, hermitian,
#                  prod, Nullable{Function}(), Nullable{Function}())


# Apply an operator to a vector.
function *(op :: AbstractLinearOperator, v :: Vector)
  size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
  op.prod(v)
end

# function *{T}(op :: AbstractLinearOperator{T}, v :: Vector{T})
#   size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
#   op.prod(v)
# end

# *{T, S}(op::AbstractLinearOperator{T}, v::Vector{S}) = op * convert(Vector{T}, v)

# function *{T, S}(op::AbstractLinearOperator{T}, v::Vector{S})
#   if S <: T
#     return op * convert(Vector{T}, v)
#   end
#   U = promote_type(T, S)
#   opU = LinearOperator{U}(size(op)..., op.symmetric, op.hermitian, op.prod, op.tprod, op.ctprod)
#   return opU * convert(Vector{U}, v)
# end


"Materialize an operator as a dense array using `op.ncol` products"
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

.*(op :: AbstractLinearOperator, x :: Number) = op * x
.*(x :: Number, op :: AbstractLinearOperator) = x * op

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
.+(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
.+(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) - op
-(op :: AbstractLinearOperator, M :: AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
.-(op :: AbstractLinearOperator, x :: Number) = op .+ (-x)
.-(x :: Number, op :: AbstractLinearOperator) = x .+ (-op)


# Utility functions.

"Cheap check that the operator and its conjugate transposed are related."
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

"Cheap check that the operator is Hermitian."
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

"Cheap check that the operator is positive (semi-)definite."
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

"Identity operator of order `n` and of data type `T`."
opEye(T :: DataType, n :: Int) = LinearOperator{T}(n, n, true, true,
                                                   v -> v[:], u -> u[:], w -> w[:])
opEye(n :: Int) = opEye(Float64, n)

"Operator of all ones of size `nrow`-by-`ncol` and of data type `T`."
opOnes(T :: DataType, nrow :: Int, ncol :: Int) = LinearOperator{T}(nrow, ncol, nrow == ncol, nrow == ncol,
                                                                    v -> sum(v) * ones(nrow),
                                                                    u -> sum(u) * ones(ncol),
                                                                    w -> sum(w) * ones(ncol))
opOnes(nrow :: Int, ncol :: Int) = opOnes(Float64, nrow, ncol)

"Zero operator of size `nrow`-by-`ncol` and of data type `T`."
opZeros(T :: DataType, nrow :: Int, ncol :: Int) = LinearOperator{T}(nrow, ncol,
                                                                     nrow == ncol, nrow == ncol,
                                                                     v -> zeros(nrow),
                                                                     u -> zeros(ncol),
                                                                     w -> zeros(ncol))
opZeros(nrow :: Int, ncol :: Int) = opZeros(Float64, nrow, ncol)

"Diagonal operator with the vector `d` on its main diagonal."
opDiagonal{T}(d :: Vector{T}) = LinearOperator{T}(length(d), length(d), true, isreal(d),
                                                  v -> v .* d,
                                                  u -> u .* d,
                                                  w -> w .* conj(d))

"""Rectangular diagonal operator of size `nrow`-by-`ncol`
with the vector `d` on its main diagonal."""
function opDiagonal{T}(nrow :: Int, ncol :: Int, d :: Vector{T})
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

function hcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end


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

function vcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end


"""Inverse of a matrix as a linear operator using `\\`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`."""
opInverse{T}(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) =
  LinearOperator{T}(size(M,2), size(M,1), symmetric, hermitian,
                    v -> M \ v, u -> M.' \ u, w -> M' \ w)

"""Inverse of a positive definite matrix as a linear operator
using its Cholesky factorization. The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks."""
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
    L = chol(M, Val{:L})
    return LinearOperator{eltype(L)}(m, m, isreal(M), true,
                                     v -> L' \ (L \ v),
                                     u -> L.' \ (conj(L \ conj(u))),
                                     w -> L' \ (L \ w))
  end
  #TODO: use iterative refinement.
end

"""Inverse of a symmetric matrix as a linear operator
using its LDL' factorization if it exists. The factorization is computed
only once. The optional `check` argument will perform a cheap hermicity
check."""
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

"""Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h h') x`."""
opHouseholder{T}(h :: Vector{T}) = LinearOperator{T}(length(h), length(h), isreal(h), true,
                                                     v -> (v - 2 * dot(h, v) * h),
                                                     Nullable{Function}(),  # Will be inferred.
                                                     w -> (w - 2 * dot(h, w) * h))



"A symmetric/hermitian operator based on the diagonal and lower triangle."
function opHermitian{S,T}(d :: Vector{S}, A :: AbstractMatrix{T})
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  return LinearOperator{U}(m, m, isreal(A), true,
                           v -> (d .* v + L * v + (v' * L)')[:],
                           Nullable{Function}(),
                           Nullable{Function}())
end


"A symmetric/hermitian operator based on a matrix."
function opHermitian(T :: AbstractMatrix)
  d = diag(T)
  return opHermitian(d, T)
end

include("qn.jl")  # quasi-Newton operators

function RestrictionOperator(I :: Vector{Int}, ncol :: Int)
  if any(I .> ncol | I .< 1)
    throw(LinearOperatorException("`I` should be a collection of index {1,…,n}, in this case, n=$ncol"))
  end
  nrow = length(I)
  tprod(x) = begin
    z = zeros(eltype(x), ncol)
    z[I] = x
    return z
  end
  return LinearOperator{Int}(nrow, ncol, false, false, x -> x[I], tprod, tprod)
end  

ExtensionOperator(I :: Vector{Int}, ncol :: Int) = RestrictionOperator(I, ncol)'

end  # module
