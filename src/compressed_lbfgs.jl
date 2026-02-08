#=
Compressed LBFGS implementation from:
    REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS
    Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994)
    DOI: 10.1007/BF01582063

Implemented by Paul Raynaud (supervised by Dominique Orban)
=#

using LinearAlgebra, LinearAlgebra.BLAS
using Requires

export CompressedLBFGSOperator, CompressedLBFGSData
# export default_matrix_type, default_vector_type

default_matrix_type(; T::DataType=Float64) = Matrix{T}
default_vector_type(; T::DataType=Float64) = Vector{T}

@init begin
  @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
    default_matrix_type(; T::DataType=Float64) = CUDA.functional() ? CUDA.CuMatrix{T, CUDA.Mem.DeviceBuffer} : Matrix{T}
    default_vector_type(; T::DataType=Float64) = CUDA.functional() ? CUDA.CuVector{T, CUDA.Mem.DeviceBuffer} : Vector{T}
  end
  # this scheme may be extended to other GPU backend modules
end

function columnshift!(A::AbstractMatrix{T}; direction::Int=-1, indicemax::Int=size(A)[1]) where T
  map(i-> view(A,:,i+direction) .= view(A,:,i), 1-direction:indicemax)
  return A
end

function vectorshift!(v::AbstractVector{T}; direction::Int=-1, indicemax::Int=length(v)) where T
  view(v, 1:indicemax+direction) .= view(v,1-direction:indicemax)
  return v
end

"""
    CompressedLBFGSData{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

A LBFGS limited-memory operator.
It represents a linear application Rⁿˣⁿ, considering at most `mem` BFGS updates.
This implementation considers the bloc matrices reoresentation of the BFGS (forward) update.
It follows the algorithm described in [REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS](https://link.springer.com/article/10.1007/BF01582063) from Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994).
This operator considers several fields directly related to the bloc representation of the operator:
- `mem`: the maximal memory of the operator;
- `n`: the dimension of the linear application;
- `k`: the current memory's size of the operator;
- `α`: scalar for `B₀ = α I`;
- `Sₖ`: retain the `k`-th last vectors `s` from the updates parametrized by `(s,y)`;
- `Yₖ`: retain the `k`-th last vectors `y` from the updates parametrized by `(s,y)`;;
- `Dₖ`: a diagonal matrix mandatory to perform the linear application and to form the matrix;
- `Lₖ`: a lower diagonal mandatory to perform the linear application and to form the matrix.
In addition to this structures which are circurlarly update when `k` reaches `mem`, we consider other intermediate data structures renew at each update:
- `chol_matrix`: a matrix required to store a Cholesky factorization of a Rᵏˣᵏ matrix;
- `intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `intermediary_vector`: a vector ∈ Rᵏ to store intermediate solutions;
- `sol`: a vector ∈ Rᵏ to store intermediate solutions;
This implementation is designed to work either on CPU or GPU.
"""
mutable struct CompressedLBFGSData{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}, I <: Integer}
  mem::Int # memory of the operator
  n::I # vector size
  k::I # k ≤ mem, active memory of the operator
  α::T # B₀ = αI
  Sₖ::M # gather all sₖ₋ₘ : n * mem
  Yₖ::M # gather all yₖ₋ₘ : n * mem
  Dₖ::Diagonal{T,V} # mem * mem
  Lₖ::LowerTriangular{T,M} # mem * mem

  chol_matrix::M # 2mem * 2mem
  intermediate_diagonal::Diagonal{T,V} # mem * mem
  intermediate_1::UpperTriangular{T,M} # 2mem * 2mem
  intermediate_2::LowerTriangular{T,M} # 2mem * 2mem
  inverse_intermediate_1::UpperTriangular{T,M} # 2mem * 2mem
  inverse_intermediate_2::LowerTriangular{T,M} # 2mem * 2mem
  intermediary_vector::V # 2mem
  sol::V # 2mem

  nprod::I
end

"""
    CompressedLBFGSData(n::Int; [T=Float64, mem=5], gpu:Bool)

A implementation of a LBFGS operator (forward), representing a `nxn` linear application.
It considers at most `k` BFGS iterates, and fit the architecture depending if it is launched on a CPU or a GPU.
"""
function CompressedLBFGSData(n::I; mem::I=5, T=Float64, M=default_matrix_type(; T), V=default_vector_type(; T)) where {I<:Integer}
  α = (T)(1)
  k = 0  
  Sₖ = M(undef, n, mem)
  Yₖ = M(undef, n, mem)
  Dₖ = Diagonal(V(undef, mem))
  Lₖ = LowerTriangular(M(undef, mem, mem))
  Lₖ.data .= zero(T)

  chol_matrix = M(undef, mem, mem)
  intermediate_diagonal = Diagonal(V(undef, mem))
  intermediate_1 = UpperTriangular(M(undef, 2*mem, 2*mem))
  intermediate_2 = LowerTriangular(M(undef, 2*mem, 2*mem))
  inverse_intermediate_1 = UpperTriangular(M(undef, 2*mem, 2*mem))
  inverse_intermediate_2 = LowerTriangular(M(undef, 2*mem, 2*mem))
  intermediary_vector = V(undef, 2*mem)
  sol = V(undef, 2*mem)

  nprod = 0

  return CompressedLBFGSData{T,M,V,I}(mem, n, k, α, Sₖ, Yₖ, Dₖ, Lₖ, chol_matrix, intermediate_diagonal, intermediate_1, intermediate_2, inverse_intermediate_1, inverse_intermediate_2, intermediary_vector, sol, nprod)
end

mutable struct CompressedLBFGSOperator{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}, F, I <: Integer}  <: AbstractQuasiNewtonOperator{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  Bv::V
  data::CompressedLBFGSData{T,M,V}
  prod!::F    # apply the operator to a vector
  tprod!::F    # apply the transpose operator to a vector
  ctprod!::F   # apply the transpose conjugate operator to a vector
end

function CompressedLBFGSOperator(n::I; mem::I=5, T=Float64, M=default_matrix_type(; T), V=default_vector_type(; T)) where {I <: Integer}
  nrow = n
  ncol = n
  symmetric = true
  hermitian = true
  Bv = V(undef, n)
  data = CompressedLBFGSData(n; mem, T, M, V)

  prod! = @closure (res, v, α, β) -> begin
    mul!(Bv, data, v)
    if β == zero(T)
      res .= α .* Bv
    else
      res .= α .* Bv .+ β .* res
    end
  end    

  F = typeof(prod!)
  
  return CompressedLBFGSOperator{T,M,V,F,I}(nrow, ncol, symmetric, hermitian, Bv, data, prod!, prod!, prod!)
end

has_args5(op::CompressedLBFGSOperator) = true
use_prod5!(op::CompressedLBFGSOperator) = true

Base.push!(op::CompressedLBFGSOperator{T,M,V}, s::V, y::V) where {T,M,V<:AbstractVector{T}} = Base.push!(op.data, s, y)
function Base.push!(data::CompressedLBFGSData{T,M,V}, s::V, y::V) where {T,M,V<:AbstractVector{T}}
  if data.k < data.mem # still some place in the structures
    data.k += 1
    view(data.Sₖ, :, data.k) .= s
    view(data.Yₖ, :, data.k) .= y
    view(data.Dₖ.diag, data.k) .= dot(s, y)
    mul!(view(data.Lₖ.data, data.k, 1:data.k-1), transpose(view(data.Yₖ, :, 1:data.k-1)), view(data.Sₖ, :, data.k) )
  else # k == mem update circurlarly the intermediary structures
    columnshift!(data.Sₖ; indicemax=data.k)
    columnshift!(data.Yₖ; indicemax=data.k)
    # data.Dₖ .= circshift(data.Dₖ, (-1, -1))
    vectorshift!(data.Dₖ.diag; indicemax=data.k)
    view(data.Sₖ, :, data.k) .= s
    view(data.Yₖ, :, data.k) .= y
    view(data.Dₖ.diag, data.k) .= dot(s, y)

    map(i-> view(data.Lₖ, i:data.mem-1, i-1) .= view(data.Lₖ, i+1:data.mem, i), 2:data.mem)
    mul!(view(data.Lₖ.data, data.k, 1:data.k-1), transpose(view(data.Yₖ, :, 1:data.k-1)), view(data.Sₖ, :, data.k) )
  end

  # step 4 and 6
  precompile_iterated_structure!(data)

  # secant equation fails if uncommented
  # data.α = dot(y,s)/dot(s,s)
  return data
end

# Algorithm 3.2 (p15)
# Theorem 2.3 (p6)
Base.Matrix(op::CompressedLBFGSOperator{T,M,V}) where {T,M,V} = Base.Matrix(op.data)
function Base.Matrix(data::CompressedLBFGSData{T,M,V}) where {T,M,V}
  B₀ = M(undef, data.n, data.n)
  map(i -> B₀[i, i] = data.α, 1:data.n)

  BSY = M(undef, data.n, 2*data.k)
  (data.k > 0) && (BSY[:, 1:data.k] = B₀ * data.Sₖ[:, 1:data.k])
  (data.k > 0) && (BSY[:, data.k+1:2*data.k] = data.Yₖ[:, 1:data.k])
  _C = M(undef, 2*data.k, 2*data.k)
  (data.k > 0) && (_C[1:data.k, 1:data.k] .= transpose(data.Sₖ[:, 1:data.k]) * data.Sₖ[:, 1:data.k])
  (data.k > 0) && (_C[1:data.k, data.k+1:2*data.k] .= data.Lₖ[1:data.k, 1:data.k])
  (data.k > 0) && (_C[data.k+1:2*data.k, 1:data.k] .= transpose(data.Lₖ[1:data.k, 1:data.k]))
  (data.k > 0) && (_C[data.k+1:2*data.k, data.k+1:2*data.k] .-= data.Dₖ[1:data.k, 1:data.k])
  C = inv(_C)

  Bₖ = B₀ .- BSY * C * transpose(BSY)
  return Bₖ
end

# Algorithm 3.2 (p15)
# step 4, Jₖ is computed only if needed
function inverse_cholesky(data::CompressedLBFGSData{T,M,V}) where {T,M,V}
  view(data.intermediate_diagonal.diag, 1:data.k) .= inv.(view(data.Dₖ.diag, 1:data.k))
  
  mul!(view(data.inverse_intermediate_1, 1:data.k, 1:data.k), view(data.intermediate_diagonal, 1:data.k, 1:data.k), transpose(view(data.Lₖ, 1:data.k, 1:data.k)))
  mul!(view(data.chol_matrix, 1:data.k, 1:data.k), view(data.Lₖ, 1:data.k, 1:data.k), view(data.inverse_intermediate_1, 1:data.k, 1:data.k))

  mul!(view(data.chol_matrix, 1:data.k, 1:data.k), transpose(view(data.Sₖ, :, 1:data.k)), view(data.Sₖ, :, 1:data.k), data.α, (T)(1))

  cholesky!(Symmetric(view(data.chol_matrix, 1:data.k, 1:data.k)))
  Jₖ = transpose(UpperTriangular(view(data.chol_matrix, 1:data.k, 1:data.k)))
  return Jₖ
end

# step 6, must be improve
function precompile_iterated_structure!(data::CompressedLBFGSData)
  Jₖ = inverse_cholesky(data)

  # constant update
  view(data.intermediate_1, data.k+1:2*data.k, 1:data.k) .= 0
  view(data.intermediate_2, 1:data.k, data.k+1:2*data.k) .= 0
  view(data.intermediate_1, data.k+1:2*data.k, data.k+1:2*data.k) .= transpose(Jₖ)
  view(data.intermediate_2, data.k+1:2*data.k, data.k+1:2*data.k) .= Jₖ

  # updates related to D^(1/2)
  view(data.intermediate_diagonal.diag, 1:data.k) .= sqrt.(view(data.Dₖ.diag, 1:data.k))
  view(data.intermediate_1, 1:data.k,1:data.k) .= .- view(data.intermediate_diagonal, 1:data.k, 1:data.k)
  view(data.intermediate_2, 1:data.k, 1:data.k) .= view(data.intermediate_diagonal, 1:data.k, 1:data.k)

  # updates related to D^(-1/2)
  view(data.intermediate_diagonal.diag, 1:data.k) .= (x -> 1/sqrt(x)).(view(data.Dₖ.diag, 1:data.k))
  mul!(view(data.intermediate_1, 1:data.k,data.k+1:2*data.k), view(data.intermediate_diagonal, 1:data.k, 1:data.k), transpose(view(data.Lₖ, 1:data.k, 1:data.k)))
  mul!(view(data.intermediate_2, data.k+1:2*data.k, 1:data.k), view(data.Lₖ, 1:data.k, 1:data.k), view(data.intermediate_diagonal, 1:data.k, 1:data.k))
  view(data.intermediate_2, data.k+1:2*data.k, 1:data.k) .= view(data.intermediate_2, data.k+1:2*data.k, 1:data.k) .* -1
  
  view(data.inverse_intermediate_1, 1:2*data.k, 1:2*data.k) .= inv(data.intermediate_1[1:2*data.k, 1:2*data.k])
  view(data.inverse_intermediate_2, 1:2*data.k, 1:2*data.k) .= inv(data.intermediate_2[1:2*data.k, 1:2*data.k])
end

# Algorithm 3.2 (p15)
LinearAlgebra.mul!(Bv::V, op::CompressedLBFGSOperator{T,M,V}, v::V) where {T,M,V<:AbstractVector{T}} = LinearAlgebra.mul!(Bv, op.data, v)
function LinearAlgebra.mul!(Bv::V, data::CompressedLBFGSData{T,M,V}, v::V) where {T,M,V<:AbstractVector{T}}
  data.nprod += 1
  # step 1-4 and 6 mainly done by Base.push!
  # step 5
  mul!(view(data.sol, 1:data.k), transpose(view(data.Yₖ, :, 1:data.k)), v)
  mul!(view(data.sol, data.k+1:2*data.k), transpose(view(data.Sₖ, :, 1:data.k)), v)
  # scal!(data.α, view(data.sol, data.k+1:2*data.k)) # more allocation, slower
  view(data.sol, data.k+1:2*data.k) .*= data.α

  mul!(view(data.intermediary_vector, 1:2*data.k), view(data.inverse_intermediate_2, 1:2*data.k, 1:2*data.k), view(data.sol, 1:2*data.k))
  mul!(view(data.sol, 1:2*data.k), view(data.inverse_intermediate_1, 1:2*data.k, 1:2*data.k), view(data.intermediary_vector, 1:2*data.k))
  
  # step 7 
  mul!(Bv, view(data.Yₖ, :, 1:data.k),  view(data.sol, 1:data.k))
  mul!(Bv, view(data.Sₖ, :, 1:data.k), view(data.sol, data.k+1:2*data.k), - data.α, (T)(-1))
  Bv .+= data.α .* v 
  return Bv
end

"""
    reset!(op)

Resets the CompressedLBFGS data of the given operator.
"""
function reset!(op::CompressedLBFGSOperator) 
  op.data.nprod = 0
  return op
end