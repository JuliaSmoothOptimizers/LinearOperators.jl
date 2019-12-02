# Constructors.
"""
    LinearOperator(M; symmetric=false, hermitian=false)

Construct a linear operator from a dense or sparse matrix.
Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
function LinearOperator(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) where T
  nrow, ncol = size(M)
  prod = @closure v -> M * v
  tprod = @closure u -> transpose(M) * u
  ctprod = @closure w -> adjoint(M) * w
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
LinearOperator(M :: SymTridiagonal{T}) where T =
  LinearOperator(M; symmetric=true, hermitian=eltype(M) <: Real)

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
LinearOperator(M :: Symmetric{T}) where T =
  LinearOperator(M; symmetric=true, hermitian=eltype(M) <: Real)

"""
    LinearOperator(M)

Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
LinearOperator(M :: Hermitian{T}) where T =
  LinearOperator(M; symmetric=eltype(M) <: Real, hermitian=true)

# the only advantage of this constructor is optional args
# use LinearOperator{Float64} if you mean real instead of complex
"""
    LinearOperator(nrow, ncol, symmetric, hermitian, prod,
                    [tprod=nothing,
                    ctprod=nothing])

Construct a linear operator from functions.
"""
function LinearOperator(nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod,
                        tprod=nothing,
                        ctprod=nothing)

  T = hermitian ? (symmetric ? Float64 : ComplexF64) : ComplexF64
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
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
function LinearOperator(::Type{T}, nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod,
                        tprod=nothing,
                        ctprod=nothing) where T

  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end
