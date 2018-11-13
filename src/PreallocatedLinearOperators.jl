export PreallocatedLinearOperator

abstract type PreallocatedAbstractLinearOperator{T,F1,F2,F3} <: AbstractLinearOperator{T,F1,F2,F3} end

"""
Type to represent a linear operator with preallocation. Implicit modifications may
happen if used without care:
```
op = PreallocatedLinearOperator(rand(5, 5))
v  = rand(5)
x = op * v        # Uses internal storage and passes pointer to x
y = op * ones(5)  # Stores on the same memory as x.
y === x           # true. op * v is lost

x = op * v        # Uses internal storage and passes pointer to x
y = op * x        # Breaks! Equivalent to mul!(x, A, x)
y == zeros(5)     # true. op * v and op * x are lost
```
"""
mutable struct PreallocatedLinearOperator{T,F1<:FuncOrNothing,F2<:FuncOrNothing,F3<:FuncOrNothing} <: PreallocatedAbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1 # apply the operator to a vector
  tprod  :: F2 # apply the transpose operator to a vector
  ctprod :: F3 # apply the transpose conjugate operator to a vector
end

"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io :: IO, op :: PreallocatedAbstractLinearOperator)
  s  = "Preallocated linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
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
function PreallocatedLinearOperator(Mv :: Vector{T}, Mtu :: Vector{T}, Maw :: Vector{T},
                                    M :: AbstractMatrix{T};
                                    symmetric=false, hermitian=false) where T
  nrow, ncol = size(M)
  @assert length(Mv) == nrow
  @assert length(Mtu) == ncol
  @assert length(Maw) == ncol
  prod = @closure v -> mul!(Mv, M, v)
  tprod = @closure u -> mul!(Mtu, transpose(M), u)
  ctprod = @closure w -> mul!(Maw, adjoint(M), w)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  PreallocatedLinearOperator{T,F1,F2,F3}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

"""
    PreallocatedLinearOperator(Mv, M; symmetric=false, hermitian=false)

Construct a linear operator of a square matrix `M` with preallocation using `Mv` as
storage space for all matrix-vector products. Notice that implicit modifications can
happen between any matrix-vector product now.
```
op = PreallocatedLinearOperator(rand(5, 5))
v  = rand(5)
x = op * v        # Uses internal storage and passes pointer to x
y = op' * ones(5) # Stores on the same memory as x, even though op is transposed
y === x           # true. op * v is lost
```
"""
PreallocatedLinearOperator(Mv :: Vector{T}, M :: AbstractMatrix{T};
                           symmetric=false, hermitian=false) where T =
  PreallocatedLinearOperator(Mv, Mv, Mv, M, symmetric=symmetric, hermitian=hermitian)

function PreallocatedLinearOperator(M :: AbstractMatrix{T};
                                    symmetric=false, hermitian=false) where T
  nrow, ncol = size(M)
  local Mv, Mtu, Maw
  if T <: Real
    if symmetric
      Maw = Mtu = Mv = Vector{T}(undef, nrow)
    else
      Mv = Vector{T}(undef, nrow)
      Maw = Mtu = Vector{T}(undef, ncol)
    end
  else
    if symmetric && hermitian # Actually real, but T is not
      Maw = Mtu = Mv = Vector{T}(undef, nrow)
    elseif symmetric
      Mtu = Mv = Vector{T}(undef, nrow)
      Maw = Vector{T}(undef, ncol)
    elseif symmetric
      Mv = Vector{T}(undef, nrow)
      Maw = Mtu = Vector{T}(undef, ncol)
    else
      Mv = Vector{T}(undef, nrow)
      Mtu = Vector{T}(undef, ncol)
      Maw = Vector{T}(undef, ncol)
    end
  end
  PreallocatedLinearOperator(Mv, Mtu, Maw, M, symmetric=symmetric, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(M :: SymTridiagonal{T}) where T
  nrow, ncol = size(M)
  Mv = Vector{T}(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : Vector{T}(undef, ncol)
  PreallocatedLinearOperator(Mv, Mv, Maw, M, symmetric=true, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(M :: Symmetric{T}) where T
  nrow, ncol = size(M)
  Mv = Vector{T}(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : Vector{T}(undef, ncol)
  PreallocatedLinearOperator(Mv, Mv, Maw, M, symmetric=true, hermitian=hermitian)
end

"""
    PreallocatedLinearOperator(M)

Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
function PreallocatedLinearOperator(M :: Hermitian{T}) where T
  nrow, ncol = size(M)
  Mv = Vector{T}(undef, nrow)
  symmetric = eltype(M) <: Real
  Mtu = symmetric ? Mv : Vector{T}(undef, ncol)
  PreallocatedLinearOperator(Mv, Mtu, Mv, M, symmetric=symmetric, hermitian=true)
end
