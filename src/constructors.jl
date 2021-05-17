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

# the only advantage of this constructor is optional args
# use LinearOperator{Float64} if you mean real instead of complex
"""
    LinearOperator(nrow, ncol, symmetric, hermitian, prod,
                    [tprod=nothing,
                    ctprod=nothing])
Construct a linear operator from functions.
"""
function LinearOperator(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod,
  tprod = nothing,
  ctprod = nothing,
) where {I<:Integer}
  T = hermitian ? (symmetric ? Float64 : ComplexF64) : ComplexF64
  Mv = Vector{T}(undef, nrow)
  Mtu = symmetric ? Mv : Vector{T}(undef, ncol)
  Maw = hermitian ? Mv : Vector{T}(undef, ncol)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, Mv, Mtu, Maw)
end

"""
    LinearOperator(type, nrow, ncol, symmetric, hermitian, prod,
                    [tprod=nothing,
                    ctprod=nothing])
Construct a linear operator from functions where the type is specified as the first argument.
Notice that the linear operator does not enforce the type, so using a wrong type can
result in errors. For instance,
```
A = [im 1.0; 0.0 1.0] # Complex matrix
op = LinearOperator(Float64, 2, 2, false, false, v->A*v, u->transpose(A)*u, w->A'*w)
Matrix(op) # InexactError
```
The error is caused because `Matrix(op)` tries to create a Float64 matrix with the
contents of the complex matrix `A`.
"""
function LinearOperator(
  ::Type{T},
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod,
  tprod = nothing,
  ctprod = nothing,
) where {T, I<:Integer}
  Mv = zeros(T, nrow)
  Mtu = symmetric ? Mv : Vector{T}(undef, ncol)
  Maw = hermitian ? Mv : Vector{T}(undef, ncol)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, Mv, Mtu, Maw)
end
