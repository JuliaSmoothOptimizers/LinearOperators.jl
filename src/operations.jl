import Base.+, Base.-, Base.*, LinearAlgebra.mul!

# Apply an operator to a vector.
function *(op :: AbstractLinearOperator{T}, v :: AbstractVector{S}) where {T,S}
  size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
  increase_nprod(op)
  op.prod(v)
end

# Unary operations.
+(op :: AbstractLinearOperator) = op

function -(op :: AbstractLinearOperator{T}) where T
  prod = @closure v -> -op.prod(v)
  tprod = @closure u -> -op.tprod(u)
  ctprod = @closure w -> -op.ctprod(w)
  LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian, prod, tprod, ctprod)
end

function mul!(y :: AbstractVector, op :: AbstractLinearOperator, x :: AbstractVector)
  y .= op * x
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
  prod = @closure v -> op1 * (op2 * v)
  tprod = @closure u -> transpose(op2) * (transpose(op1) * u)
  ctprod = @closure w -> op2' * (op1' * w)
  LinearOperator{S}(m1, n2, false, false, prod, tprod, ctprod)
end

## Matrix times operator.
*(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) * op
*(op :: AbstractLinearOperator, M :: AbstractMatrix) = op * LinearOperator(M)

## Scalar times operator.
function *(op :: AbstractLinearOperator, x :: Number)
  S = promote_type(eltype(op), typeof(x))
  prod = @closure v -> (op * v) * x
  tprod = @closure u -> x * (transpose(op) * u)
  ctprod = @closure w -> x' * (op' * w)
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod, tprod, ctprod)
end

function *(x :: Number, op :: AbstractLinearOperator)
  S = promote_type(eltype(op), typeof(x))
  prod = @closure v -> x * (op * v)
  tprod = @closure u -> (transpose(op) * u) * x
  ctprod = @closure w -> (op' * w) * x'
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod, tprod, ctprod)
end

# Operator + operator.
function +(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod = @closure v -> (op1   * v) + (op2   * v)
  tprod = @closure u -> (transpose(op1) * u) + (transpose(op2) * u)
  ctprod = @closure w -> (op1'  * w) + (op2'  * w)
  return LinearOperator{S}(m1, n1, symmetric(op1) && symmetric(op2), hermitian(op1) && hermitian(op2),
                           prod, tprod, ctprod)
end

# Operator + matrix.
+(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) + op
+(op :: AbstractLinearOperator, M :: AbstractMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
+(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
+(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) - op
-(op :: AbstractLinearOperator, M :: AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
-(op :: AbstractLinearOperator, x :: Number) = op + (-x)
-(x :: Number, op :: AbstractLinearOperator) = x + (-op)
