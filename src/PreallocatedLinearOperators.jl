export PreallocatedLinearOperator

abstract type AbstractPreallocatedLinearOperator{T} <: AbstractLinearOperator{T} end

"""
Type to represent a linear operator with preallocation. Implicit modifications may
happen if used without care:
```
op = PreallocatedLinearOperator(rand(5, 5))
v  = rand(5)
x = op * v        # Uses internal storage and passes pointer to x
y = op * ones(5)  # Overwrites the same memory as x.
y === x           # true. op * v is lost

x = op * v        # Uses internal storage and passes pointer to x
y = op * x        # Silently overwrite x to zeros! Equivalent to mul!(x, A, x).
y == zeros(5)     # true. op * v and op * x are lost
```
"""
mutable struct PreallocatedLinearOperator{T} <: AbstractPreallocatedLinearOperator{T}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod    # apply the operator to a vector
  tprod   # apply the transpose operator to a vector
  ctprod  # apply the transpose conjugate operator to a vector
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv
  Mcv
  Mtu
  Maw
end

PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod) where T =
  PreallocatedLinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, 0, 0, 0,
                                Vector{T}(undef, nrow), Vector{T}(undef, nrow),
                                Vector{T}(undef, ncol), Vector{T}(undef, ncol))

"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io :: IO, op :: AbstractPreallocatedLinearOperator)
  s  = "Preallocated linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  nprod:   %d\n", nprod(op))
  s *= @sprintf("  ntprod:  %d\n", ntprod(op))
  s *= @sprintf("  nctprod: %d\n", nctprod(op))
  s *= "\n"
  print(io, s)
end

# Constructors.
"""
    PreallocatedLinearOperator(Mv, Mtu, Maw, M; symmetric=false, hermitian=false)

Construct a linear operator from a dense or sparse matrix, using `Mv` as storage space
for `M * v` and `Mtu` as storage space for `transpose(M) * u` and `Maw` to store
`adjoint(M) * w`. Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
function PreallocatedLinearOperator(Mv :: AbstractVector{T}, Mcv :: AbstractVector{T},
                                    Mtu :: AbstractVector{T}, Maw :: AbstractVector{T},
                                    M :: AbstractMatrix{T}; symmetric=false, hermitian=false) where T
  nrow, ncol = size(M)
  @assert length(Mv) == nrow
  @assert length(Mcv) == nrow
  @assert length(Mtu) == ncol
  @assert length(Maw) == ncol
  prod = @closure v -> mul!(Mv, M, v)
  tprod = @closure u -> mul!(Mtu, transpose(M), u)
  ctprod = @closure w -> mul!(Maw, adjoint(M), w)
  PreallocatedLinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, 0, 0, 0, Mv, Mcv, Mtu, Maw)
end

function PreallocatedLinearOperator(M :: AbstractMatrix{T}; storagetype=Vector{T}, symmetric=false, hermitian=false) where T
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  Mcv = storagetype(undef, nrow)
  Mtu = storagetype(undef, ncol)
  Maw = storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mcv, Mtu, Maw, M, symmetric=symmetric, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(M :: SymTridiagonal{T}; storagetype=Vector{T}, hermitian=false) where T
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  Mcv = storagetype(undef, nrow)
  Mtu = storagetype(undef, ncol)
  Maw = storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mcv, Mtu, Maw, M, symmetric=true, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(M :: Symmetric{T}; storagetype=Vector{T}, hermitian=false) where T
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  Mcv = storagetype(undef, nrow)
  Mtu = storagetype(undef, ncol)
  Maw = storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mcv, Mtu, Maw, M, symmetric=true, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
function PreallocatedLinearOperator(M :: Hermitian{T}; storagetype=Vector{T}, symmetric=false) where T
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  Mcv = storagetype(undef, nrow)
  symmetric = eltype(M) <: Real
  Mtu = symmetric ? Mv : storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mcv, Mtu, Mv, M, symmetric=symmetric, hermitian=true)
end

function mul!(y :: AbstractVector, A :: PreallocatedLinearOperator, x :: AbstractVector)
  A.prod(x)
  increase_nprod(A)
  y .= A.Mv
  return y
end

function *(op :: PreallocatedLinearOperator{T}, v :: AbstractVector{T}) where T
  size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
  op.prod(v)
  increase_nprod(op)
  return op.Mv
end

function *(op :: AdjointLinearOperator{T,PreallocatedLinearOperator{T}}, v :: AbstractVector{T}) where T
  p = op.parent
  ishermitian(p) && return p * v
  length(v) == size(p, 1) || throw(LinearOperatorException("shape mismatch"))
  p.ctprod(v)
  return p.Maw
  if p.ctprod !== nothing
    increase_nctprod(p)
    p.ctprod(v)
    return p.Maw
  end
  increment_tprod = true
  if p.tprod === nothing
    if issymmetric(p)
      increment_tprod = false
    else
      throw(LinearOperatorException("unable to infer conjugate transpose operator"))
    end
  end
  p.Mcv .= conj.(v)
  if increment_tprod
    p.tprod(p.Mcv)
    increase_ntprod(p)
    p.Mtu .= conj.(p.Mtu)
    return p.Mtu
  else
    p.prod(p.Mcv)
    increase_nprod(p)
    p.Mv .= conj.(p.Mv)
    return p.Mv
  end
end

function *(op :: TransposeLinearOperator{T,PreallocatedLinearOperator{T}}, v :: AbstractVector{T}) where T
  p = op.parent
  issymmetric(p) && return p * v
  length(v) == size(p, 1) || throw(LinearOperatorException("shape mismatch"))
  if p.tprod !== nothing
    increase_ntprod(p)
    p.tprod(v)
    return p.Mtu
  end
  increment_ctprod = true
  if p.ctprod === nothing
    if ishermitian(p)
      increment_ctprod = false
    else
      throw(LinearOperatorException("unable to infer transpose operator"))
    end
  end
  p.Mcv .= conj.(v)
  if increment_ctprod
    p.ctprod(p.Mcv)
    increase_nctprod(p)
    p.Maw .= conj.(p.Maw)
    return p.Maw
  else
    p.prod(p.Mcv)
    increase_nprod(p)
    p.Mv .= conj.(p.Mv)
    return p.Mv
  end
end

function *(op :: ConjugateLinearOperator{T,PreallocatedLinearOperator{T}}, v :: AbstractVector{T}) where T
  p = op.parent
  p.Maw .= conj.(v)
  p.prod(p.Maw)
  p.Mv .= conj.(p.Mv)
  return p.Mv
end
