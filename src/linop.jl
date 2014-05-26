# Linear Operators for Julia
module linop

export LinearOperator, opEye, opOnes, opZeros, opDiagonal,
       opCholesky, opHouseholder

KindOfMatrix = Union(Array, SparseMatrixCSC)
FuncOrNothing = Union(Function, Nothing)

type LinearOperator
  nrow   :: Int
  ncol   :: Int
  dtype   :: DataType
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function       # apply the operator to a vector
  tprod  :: FuncOrNothing  # apply the transpose operator to a vector
  ctprod :: FuncOrNothing  # apply the transpose conjugate operator to a vector
end


import Base.size
size(op :: LinearOperator) = (op.nrow, op.ncol)
shape(op :: LinearOperator) = size(op)
hermitian(op :: LinearOperator) = op.hermitian
symmetric(op :: LinearOperator) = op.symmetric


import Base.show
function show(io :: IO, op :: LinearOperator)
  s  = "Linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  dtype: %s\n", op.dtype)
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  prod:   %s\n", string(op.prod))
  s *= @sprintf("  tprod:  %s\n", string(op.tprod))
  s *= @sprintf("  ctprod: %s", string(op.ctprod))
  print(io, s)
end


# Constructors.
LinearOperator(M :: KindOfMatrix; symmetric=false, hermitian=false) =
  LinearOperator(size(M,1), size(M,2), typeof(M[1,1]), symmetric, hermitian,
                 v -> M * v, u -> M.' * u, w -> M' * w)


# Apply an operator to a vector.
function (*)(op :: LinearOperator, v :: Vector)
  (m, n) = size(op)
  if size(v,1) != n
    error("Shape mismatch")
  end
  return op.prod(v)
end

import Base.full
function full(op :: LinearOperator)
  (m, n) = size(op)
  A = zeros(op.dtype, m, n)  # Must be of same dtype as operator.
  e = zeros(op.dtype, n)
  for i = 1 : n
    e[i] = 1;
    A[:,i] = op * e;
    e[i] = 0;
  end
  return A
end


# Unary operations.
(+)(op :: LinearOperator) = op
(-)(op :: LinearOperator) = LinearOperator(op.nrow, op.ncol, op.dtype,
                                           op.symmetric, op.hermitian,
                                           v -> -op.prod(v),
                                           u -> -op.tprod(u),
                                           w -> -op.ctprod(w))

function transpose(op :: LinearOperator)
  if op.symmetric
    return op
  end
  if op.tprod != nothing
    return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                          op.tprod, op.prod, v -> conj(op.tprod(v)))
  end
  if op.ctprod == nothing
    error("Unable to infer transpose operator")
  end

  return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(op.ctprod(conj(v))),  # A.'v = conj(A' conj(v))
                        op.prod,                        # (A.').' = A
                        w -> conj(op.prod(v)))          # (A.')' = conj(A)
end

function ctranspose(op :: LinearOperator)
  if op.hermitian
    return op
  end
  if op.ctprod != nothing
    return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                          op.ctprod, u -> conj(op.prod(u)), op.prod)
  end
  if op.tprod == nothing
    error("Unable to infer conjugate transpose operator")
  end

  return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(op.tprod(v)), u -> conj(op.prod(u)), op.prod)
end

import Base.conj
function conj(op :: LinearOperator)
  return LinearOperator(op.nrow, op.ncol, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(op.prod(conj(v))),
                        u -> op.ctprod(u),
                        w -> op.tprod(w))
end

# Binary operations.

## Operator times operator.
function (*)(op1 :: LinearOperator, op2 :: LinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    error("Shape mismatch")
  end
  result_type = promote_type(op1.dtype, op2.dtype)
  return LinearOperator(m1, n2, result_type, false, false,
                        v -> op1 * (op2 * v),
                        u -> op2.' * (op1.' * u),
                        w -> op2' * (op1' * w))
end

## Matrix times operator.
(*)(M :: KindOfMatrix, op :: LinearOperator) = LinearOperator(M) * op
(*)(op :: LinearOperator, M :: KindOfMatrix) = op * LinearOperator(M)

## Scalar times operator.
(*)(op :: LinearOperator, x :: Number) = LinearOperator(op.nrow, op.ncol,
                                                        promote_type(op.dtype, typeof(x)),
                                                        op.symmetric,
                                                        op.hermitian && isreal(x),
                                                        v -> (op * v) * x,
                                                        u -> x * (op.' * u),
                                                        w -> x' * (op' * w))
(*)(x :: Number, op :: LinearOperator) = LinearOperator(op.nrow, op.ncol,
                                                        promote_type(op.dtype, typeof(x)),
                                                        op.symmetric,
                                                        op.hermitian && isreal(x),
                                                        v -> x * (op * v),
                                                        u -> (op.' * u) * x,
                                                        w -> (op' * w) * x')
(.*)(op :: LinearOperator, x :: Number) = op * x
(.*)(x :: Number, op :: LinearOperator) = x * op

# Operator + operator.
function (+)(op1 :: LinearOperator, op2 :: LinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    error("Shape mismatch")
  end
  return LinearOperator(m1, n1, promote_type(op1.dtype, op2.dtype),
                        op1.symmetric && op2.symmetric,
                        op1.hermitian && op2.hermitian,
                        v -> (op1   * v) + (op2   * v),
                        u -> (op1.' * u) + (op2.' * u),
                        w -> (op1'  * w) + (op2'  * w))
end

# Operator + matrix.
(+)(M :: KindOfMatrix, op :: LinearOperator) = LinearOperator(M) + op
(+)(op :: LinearOperator, M :: KindOfMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
(.+)(op :: LinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
(.+)(x :: Number, op :: LinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
(-)(op1 :: LinearOperator, op2 :: LinearOperator) = op1 + (-op2)

# Operator - matrix.
(-)(M :: KindOfMatrix, op :: LinearOperator) = LinearOperator(M) - op
(-)(op :: LinearOperator, M :: KindOfMatrix) = op - LinearOperator(M)

# Operator - scalar.
(.-)(op :: LinearOperator, x :: Number) = op .+ (-x)
(.-)(x :: Number, op :: LinearOperator) = x .+ (-op)

# Special linear operators.

## Identity operator.
opEye(n :: Int; dtype=Float64) = LinearOperator(n, n, true, true, dtype,
                                                v -> v, u -> u, w -> w)

## All ones.
opOnes(nrow, ncol; dtype=Float64) = LinearOperator(nrow, ncol, dtype,
                                                  nrow == ncol, nrow == ncol,
                                                  v -> sum(v) * ones(nrow),
                                                  u -> sum(u) * ones(ncol),
                                                  w -> sum(w) * ones(ncol))

## All zeros.
opZeros(nrow, ncol; dtype=Float64) = LinearOperator(nrow, ncol, dtype,
                                                   nrow == ncol, nrow == ncol,
                                                   v -> zeros(nrow),
                                                   u -> zeros(ncol),
                                                   w -> zeros(ncol))

## Diagonal.
opDiagonal(d :: Vector) = LinearOperator(length(d), length(d), typeof(d[1]),
                                         true, !(typeof(d[1]) <: Complex),
                                         v -> v .* d,
                                         u -> u .* d,
                                         w -> w .* conj(d))

## Inverse as a Cholesky factorization.
function opCholesky(M :: KindOfMatrix; check=false)
  (m, n) = size(M)
  if m != n
    error("Shape mismatch")
  end
  if check
    # Cheap hermicity check.
    v = rand(n)
    w = M * v
    s = dot(w, w)  # = (Mv)'(Mv) = v' M' M v.
    y = M * w
    t = dot(v, y)  # = v' M M v.
    u = eps(Float64)
    if abs(s-t) > (s * u) * u^(1/3)
      error("Matrix is not symmetric")
    end
    # Cheap positive definiteness check.
    if dot(v, w) <= 0
      error("Matrix is not positive definite")
    end
  end
  L = issparse(M) ? cholfact(M) : chol(M, :L);
  return LinearOperator(m, m, typeof(M[1,1]),
                        !(typeof(M[1,1]) <: Complex), true,
                        v -> L' \ (L \ v),
                        u -> L.' \ (conj(L \ conj(u))),
                        w -> L' \ (L \ w))
  # Todo: use iterative refinement.
end

## Apply a Householder transformation stored in the vector h.
## The result is x -> (I - 2 h h') x.
opHouseholder(h :: Vector) = LinearOperator(length(h), length(h), typeof(h[1]),
                                            !(typeof(h[1]) <: Complex), true,
                                            v -> (v - 2 * dot(h, v) * h),
                                            Nothing(),  # Will be inferred.
                                            w -> (w - 2 * dot(h, w) * h))

end  # module
