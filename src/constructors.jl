# Constructors.
"""
    LinearOperator(Mv, Mtu, Maw, M; symmetric=false, hermitian=false)
Construct a linear operator from a dense or sparse matrix, using `Mv` as storage space
for `M * v` and `Mtu` as storage space for `transpose(M) * u` and `Maw` to store
`adjoint(M) * w`. Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
function LinearOperator(
  Mv::AbstractVector{T},
  Mtu::AbstractVector{T},
  Maw::AbstractVector{T},
  M::AbstractMatrix{T};
  symmetric = false,
  hermitian = false,
) where {T}
  nrow, ncol = size(M)
  @assert length(Mv) == nrow
  @assert length(Mtu) == ncol
  @assert length(Maw) == ncol
  prod! = @closure (res, v, α, β) -> mul!(res, M, v, α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(M), u, α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(M), w, α, β)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, Mv, Mtu, Maw)
end

  """
  LinearOperator(Mv, M :: Symmetric{<:Real})

Construct a linear operator from a symmetric real square matrix `M` with preallocation
using `Mv` as storage space.
"""
LinearOperator(
  Mv::AbstractVector{T},
  M::Union{SymTridiagonal{T}, Symmetric{T}},
  ) where {T <: Real} = LinearOperator(Mv, Mv, Mv, M, symmetric = true, hermitian = true)

  """
    LinearOperator(M)

Constructs a linear operator from a matrix.
"""
function LinearOperator(M::AbstractMatrix{T}; storagetype = Vector{T}, symmetric = false, hermitian = false) where {T}
  nrow, ncol = size(M)
  local Mv, Mtu, Maw
  if T <: Real
    if symmetric
      Maw = Mtu = Mv = storagetype(undef, nrow)
    else
      Mv = storagetype(undef, nrow)
      Maw = Mtu = storagetype(undef, ncol)
    end
  else
    if symmetric && hermitian # Actually real, but T is not
      Maw = Mtu = Mv = storagetype(undef, nrow)
    elseif symmetric
      Mtu = Mv = storagetype(undef, nrow)
      Maw = storagetype(undef, ncol)
    elseif hermitian
      Mv = storagetype(undef, nrow)
      Maw = Mtu = storagetype(undef, ncol)
    else
      Mv = storagetype(undef, nrow)
      Mtu = storagetype(undef, ncol)
      Maw = storagetype(undef, ncol)
    end
  end
  LinearOperator(Mv, Mtu, Maw, M, symmetric = symmetric, hermitian = hermitian)
end

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function LinearOperator(M::SymTridiagonal{T}; storagetype = Vector{T}, kwargs...) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : storagetype(undef, ncol)
  LinearOperator(Mv, Mv, Maw, M, symmetric = true, hermitian = hermitian)
end

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function LinearOperator(M::Symmetric{T}; storagetype = Vector{T}, kwargs...) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : storagetype(undef, ncol)
  LinearOperator(Mv, Mv, Maw, M, symmetric = true, hermitian = hermitian)
end

"""
    LinearOperator(M)
    
Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
function LinearOperator(M::Hermitian{T}; storagetype = Vector{T}, kwargs...) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  symmetric = eltype(M) <: Real
  Mtu = symmetric ? Mv : storagetype(undef, ncol)
  LinearOperator(Mv, Mtu, Mv, M, symmetric = symmetric, hermitian = true)
end

"""
    LinearOperator(type, nrow, ncol, symmetric, hermitian, prod!,
                    [tprod!=nothing,
                    ctprod!=nothing])
                    
Construct a linear operator from functions where the type is specified as the first argument.
Notice that the linear operator does not enforce the type, so using a wrong type can
result in errors. For instance,
```
A = [im 1.0; 0.0 1.0] # Complex matrix
function mulOp!(res, M, v, α, β)
  mul!(res, M, v, α, β)
end
op = LinearOperator(Float64, 2, 2, false, false, 
                    (res, v, α, β) -> mulOp!(res, A, v, α, β), 
                    (res, u, α, β) -> mulOp!(res, transpose(A), u, α, β), 
                    (res, w, α, β) -> mulOp!(res, A', w, α, β))
Matrix(op) # InexactError
```
The error is caused because `Matrix(op)` tries to create a Float64 matrix with the
contents of the complex matrix `A`.

You will have to be carefull because using `*` may generate a vector that contains `NaN` values.
This can also append if you use the 3-args `mul!` function with a preallocated vector such as 
`Vector{Float64}(undef, n)`.
To fix this issue you will have to deal with the cases `β == 0` and `β != 0` separately:
```
d1 = [2.0; 3.0]
function mulSquareOpDiagonal!(res, d, v, α, β::T) where T
  if β == zero(T)
    res .= α .* d .* v
  else 
    res .= α .* d .* v .+ β .* res
  end
end
op = LinearOperator(Float64, 2, 2, true, true, 
                    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β))
```
"""
function LinearOperator(
  ::Type{T},
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod! = nothing,
  ctprod! = nothing,
) where {T, I<:Integer}
  Mv = zeros(T, nrow)
  Mtu = symmetric ? Mv : Vector{T}(undef, ncol)
  Maw = hermitian ? Mv : Vector{T}(undef, ncol)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, Mv, Mtu, Maw)
end
