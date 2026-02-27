import Base.+, Base.-, Base.*, Base./, LinearAlgebra.mul!

function allocate_vectors_args3!(op::AbstractLinearOperator)
  S = storage_type(op)
  op.Mv5 = S(undef, op.nrow)
  op.Mtu5 = (op.nrow == op.ncol) ? op.Mv5 : S(undef, op.ncol)
  op.allocated5 = true
end

function prod3!(res, prod!, v, α, β, Mv5)
  if β == 0
    prod!(res, v)
    if α != 1
      res .*= α
    end
  else
    prod!(Mv5, v)
    res .= α .* Mv5 .+ β .* res
  end
end

function mul!(res::AbstractVector, op::AbstractLinearOperator{T}, v::AbstractVector, α, β) where {T}
  use_p5! = use_prod5!(op)
  has_args5(op) || (β == 0) || isallocated5(op) || allocate_vectors_args3!(op)
  (size(v, 1) == size(op, 2) && size(res, 1) == size(op, 1)) ||
    throw(LinearOperatorException("shape mismatch"))
  increase_nprod!(op)
  if use_p5!
    op.prod!(res, v, α, β)
  else
    prod3!(res, op.prod!, v, α, β, op.Mv5)
  end
end

function mul!(res::AbstractMatrix, op::AbstractLinearOperator, m::AbstractMatrix{T}, α, β) where {T}
  op.prod!(res, m, α, β)
end

function mul!(res::AbstractVecOrMat, op::AbstractLinearOperator, v::AbstractVecOrMat{T}) where {T}
  mul!(res, op, v, one(T), zero(T))
end

# Apply an operator to a vector.
function *(op::AbstractLinearOperator{T}, v::AbstractVector{S}) where {T, S}
  nrow, ncol = size(op)
  res = similar(v, promote_type(T, S), nrow)
  mul!(res, op, v)
  return res
end

function mul!(
  res::Adjoint{S1, V1},
  v::Adjoint{S2, V2},
  op::AbstractLinearOperator{T},
) where {T, S1, S2, V1 <: AbstractVector{S1}, V2 <: AbstractVector{S2}}
  mul!(adjoint(res), adjoint(op), adjoint(v))
  return res
end

function mul!(
  res::Transpose{S1, V1},
  v::Transpose{S2, V2},
  op::AbstractLinearOperator{T},
) where {T, S1, S2, V1 <: AbstractVector{S1}, V2 <: AbstractVector{S2}}
  mul!(transpose(res), transpose(op), transpose(v))
  return res
end

function *(
  v::Union{Transpose{S, V}, Adjoint{S, V}},
  op::AbstractLinearOperator{T},
) where {T, S, V <: AbstractVector{S}}
  nrow, ncol = size(op)
  res = similar(v.parent, promote_type(T, S), ncol)
  v_wrapper = typeof(v).name.wrapper
  mul!(v_wrapper(res), v, op)
  return v_wrapper(res)
end

# Apply an operator to a matrix (only in-place, since operator * matrix is a matrix).

function mul!(
  res::Adjoint{S1, M1},
  m::Adjoint{S2, M2},
  op::AbstractLinearOperator{T},
) where {T, S1, S2, M1 <: AbstractMatrix{S1}, M2 <: AbstractMatrix{S2}}
  mul!(adjoint(res), adjoint(op), adjoint(m))
  return res
end

function mul!(
  res::Transpose{S1, M1},
  m::Transpose{S2, M2},
  op::AbstractLinearOperator{T},
) where {T, S1, S2, M1 <: AbstractMatrix{S1}, M2 <: AbstractMatrix{S2}}
  mul!(transpose(res), transpose(op), transpose(m))
  return res
end

# Unary operations.
+(op::AbstractLinearOperator) = op

function -(op::AbstractLinearOperator{T}) where {T}
  prod! = @closure (res, v, α, β) -> mul!(res, op, v, -α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(op), u, -α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(op), w, -α, β)
  CompositeLinearOperator(
    T,
    op.nrow,
    op.ncol,
    op.symmetric,
    op.hermitian,
    prod!,
    tprod!,
    ctprod!,
    has_args5(op),
    S = storage_type(op),
  )
end

function prod_op!(
  res::AbstractVecOrMat,
  op1::AbstractLinearOperator,
  op2::AbstractLinearOperator,
  vtmp::AbstractVecOrMat,
  v::AbstractVecOrMat,
  α,
  β,
)
  mul!(vtmp, op2, v)
  mul!(res, op1, vtmp, α, β)
end

## Operator times operator.
function *(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  T = promote_type(eltype(op1), eltype(op2))
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(storage_type(op1), storage_type(op2))
  if !isconcretetype(S)
    throw(
      LinearOperatorException(
        "storage types $(storage_type(op1)) and $(storage_type(op2)) " *
        "cannot be promoted to a concrete type. " *
        "Ensure both operators use compatible storage types (e.g., both GPU or both CPU).",
      ),
    )
  end
  #tmp vector for products
  vtmp = fill!(S(undef, m2), zero(T))
  utmp = fill!(S(undef, n1), zero(T))
  wtmp = fill!(S(undef, n1), zero(T))
  prod! = @closure (res, v, α, β) -> prod_op!(res, op1, op2, vtmp, v, α, β)
  tprod! = @closure (res, u, α, β) -> prod_op!(res, transpose(op2), transpose(op1), utmp, u, α, β)
  ctprod! = @closure (res, w, α, β) -> prod_op!(res, adjoint(op2), adjoint(op1), wtmp, w, α, β)
  args5 = (has_args5(op1) && has_args5(op2))
  CompositeLinearOperator(T, m1, n2, false, false, prod!, tprod!, ctprod!, args5, S)
end

## Matrix times operator.
*(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) * op
*(op::AbstractLinearOperator, M::AbstractMatrix) = op * LinearOperator(M)

## Scalar times operator. (# commutation α*v ???)
function *(op::AbstractLinearOperator, x::Number)
  T = promote_type(eltype(op), typeof(x))
  prod! = @closure (res, v, α, β) -> mul!(res, op, v, x * α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(op), u, x * α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(op), w, x' * α, β)
  CompositeLinearOperator(
    T,
    op.nrow,
    op.ncol,
    op.symmetric,
    op.hermitian && isreal(x),
    prod!,
    tprod!,
    ctprod!,
    has_args5(op),
    S = storage_type(op),
  )
end

function *(x::Number, op::AbstractLinearOperator)
  return op * x
end

/(op::AbstractLinearOperator{T}, x::Number) where {T} = op * (one(T) / x)

# Operator + operator.

function sum_prod!(
  res::AbstractVecOrMat,
  op1::AbstractLinearOperator,
  op2::AbstractLinearOperator{T},
  v::AbstractVecOrMat,
  α,
  β,
) where {T}
  mul!(res, op1, v, α, β)
  mul!(res, op2, v, α, one(T))
end

function +(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  T = promote_type(eltype(op1), eltype(op2))
  prod! = @closure (res, v, α, β) -> sum_prod!(res, op1, op2, v, α, β)
  tprod! = @closure (res, u, α, β) -> sum_prod!(res, transpose(op1), transpose(op2), u, α, β)
  ctprod! = @closure (res, w, α, β) -> sum_prod!(res, adjoint(op1), adjoint(op2), w, α, β)
  symm = (issymmetric(op1) && issymmetric(op2))
  herm = (ishermitian(op1) && ishermitian(op2))
  args5 = (has_args5(op1) && has_args5(op2))
  S = promote_type(storage_type(op1), storage_type(op2))
  if !isconcretetype(S)
    throw(
      LinearOperatorException(
        "storage types $(storage_type(op1)) and $(storage_type(op2)) " *
        "cannot be promoted to a concrete type. " *
        "Ensure both operators use compatible storage types (e.g., both GPU or both CPU).",
      ),
    )
  end
  return CompositeLinearOperator(T, m1, n1, symm, herm, prod!, tprod!, ctprod!, args5, S)
end

# Operator + matrix.
+(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) + op
+(op::AbstractLinearOperator, M::AbstractMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
+(op::AbstractLinearOperator, x::Number) = op + x * opOnes(op.nrow, op.ncol)
+(x::Number, op::AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1::AbstractLinearOperator, op2::AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) - op
-(op::AbstractLinearOperator, M::AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
-(op::AbstractLinearOperator, x::Number) = op + (-x)
-(x::Number, op::AbstractLinearOperator) = x + (-op)
