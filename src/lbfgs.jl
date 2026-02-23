export LBFGSOperator, InverseLBFGSOperator, diag, diag!

"A data type to hold information relative to LBFGS operators."
mutable struct LBFGSData{T, I <: Integer}
  const mem::I
  const scaling::Bool
  scaling_factor::T
  const damped::Bool
  σ₂::T
  σ₃::T
  opnorm_upper_bound::T # Upper bound for the operator norm ‖Bₖ‖₂ ≤ ‖B₀‖₂ + ∑ᵢ ‖bᵢ‖₂²
  const s::Vector{Vector{T}}
  const y::Vector{Vector{T}}
  const ys::Vector{T}
  const α::Vector{T}
  const a::Vector{Vector{T}}
  const b::Vector{Vector{T}}
  const norm_b::Vector{T}
  insert::I
  const Ax::Vector{T}
  const shifted_p::Matrix{T} # Temporary matrix used in the computation solve_shifted_system!
  const shifted_v::Vector{T}
  const shifted_u::Vector{T}
end

function LBFGSData(
  T::Type,
  n::I;
  mem::I = 5,
  scaling::Bool = true,
  damped::Bool = false,
  inverse::Bool = true,
  σ₂::Float64 = 0.99,
  σ₃::Float64 = 10.0,
) where {I <: Integer}
  LBFGSData{T, I}(
    max(mem, 1),
    scaling,
    convert(T, 1),
    damped,
    convert(T, σ₂),
    convert(T, σ₃),
    convert(T, 1),
    [zeros(T, n) for _ = 1:mem],
    [zeros(T, n) for _ = 1:mem],
    zeros(T, mem),
    inverse ? zeros(T, mem) : zeros(T, 0),
    inverse ? Vector{T}(undef, 0) : [zeros(T, n) for _ = 1:mem],
    inverse ? Vector{T}(undef, 0) : [zeros(T, n) for _ = 1:mem],
    inverse ? Vector{T}(undef, 0) : zeros(T, mem),
    1,
    Vector{T}(undef, n),
    Array{T}(undef, (n, 2 * mem)),
    Vector{T}(undef, 2 * mem),
    Vector{T}(undef, n),
  )
end

LBFGSData(n::I; kwargs...) where {I <: Integer} = LBFGSData(Float64, n; kwargs...)

"A type for limited-memory BFGS approximations."
mutable struct LBFGSOperator{T, I <: Integer, F, Ft, Fct} <: AbstractQuasiNewtonOperator{T}
  const nrow::I
  const ncol::I
  const symmetric::Bool
  const hermitian::Bool
  const prod!::F    # apply the operator to a vector
  const tprod!::Ft    # apply the transpose operator to a vector
  const ctprod!::Fct   # apply the transpose conjugate operator to a vector
  const inverse::Bool
  const data::LBFGSData{T, I}
  nprod::I
  ntprod::I
  nctprod::I
end

LBFGSOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  inverse::Bool,
  data::LBFGSData{T, I},
) where {T, I <: Integer, F, Ft, Fct} = LBFGSOperator{T, I, F, Ft, Fct}(
  nrow,
  ncol,
  symmetric,
  hermitian,
  prod!,
  tprod!,
  ctprod!,
  inverse,
  data,
  0,
  0,
  0,
)

has_args5(op::LBFGSOperator) = true
isallocated5(op::LBFGSOperator) = true
storage_type(op::LBFGSOperator{T}) where {T} = Vector{T}

"""
    InverseLBFGSOperator(T, n, [mem=5; scaling=true])
    InverseLBFGSOperator(n, [mem=5; scaling=true])
Construct a limited-memory BFGS approximation in inverse form. If the type `T`
is omitted, then `Float64` is used.
"""
function InverseLBFGSOperator(T::Type, n::I; kwargs...) where {I <: Integer}
  kwargs = Dict(kwargs)
  delete!(kwargs, :inverse)
  lbfgs_data = LBFGSData(T, n; inverse = true, kwargs...)

  function lbfgs_multiply(
    res::AbstractVector,
    data::LBFGSData,
    x::AbstractArray,
    αm,
    βm::T2,
  ) where {T2}
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.4, p. 178.

    q = data.Ax # tmp vector
    q .= x

    @inbounds for i = 1:(data.mem)
      k = mod(data.insert - i - 1, data.mem) + 1
      if data.ys[k] != 0
        αk = dot(data.s[k], q) / data.ys[k]
        data.α[k] = αk
        q .-= αk .* data.y[k]
      end
    end

    data.scaling && (q .*= data.scaling_factor)

    @inbounds for i = 1:(data.mem)
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        αk = data.α[k]
        β = αk - dot(data.y[k], q) / data.ys[k]
        q .+= β .* data.s[k]
      end
    end
    if βm == zero(T2)
      res .= αm .* q
    else
      res .= αm .* q .+ βm .* res
    end
  end

  prod! = @closure (res, x, α, β) -> lbfgs_multiply(res, lbfgs_data, x, α, β)
  return LBFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, true, lbfgs_data)
end

InverseLBFGSOperator(n::Int; kwargs...) = InverseLBFGSOperator(Float64, n; kwargs...)

"""
    LBFGSOperator(T, n; [mem=5, scaling=true])
    LBFGSOperator(n; [mem=5, scaling=true])
Construct a limited-memory BFGS approximation in forward form. If the type `T`
is omitted, then `Float64` is used.
"""
function LBFGSOperator(T::Type, n::I; kwargs...) where {I <: Integer}
  kwargs = Dict(kwargs)
  delete!(kwargs, :inverse)
  lbfgs_data = LBFGSData(T, n; inverse = false, kwargs...)

  function lbfgs_multiply(
    res::AbstractVector,
    data::LBFGSData,
    x::AbstractArray,
    α,
    β::T2,
  ) where {T2}
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.6, p. 184.

    q = data.Ax
    q .= x

    data.scaling && (q ./= data.scaling_factor)

    # B = B₀ + Σᵢ (bᵢbᵢ' - aᵢaᵢ').
    @inbounds for i = 1:(data.mem)
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        ax = dot(data.a[k], x)
        bx = dot(data.b[k], x)
        q .+= bx .* data.b[k] .- ax .* data.a[k]
      end
    end
    if β == zero(T2)
      res .= α .* q
    else
      res .= α .* q .+ β .* res
    end
  end

  prod! = @closure (res, x, α, β) -> lbfgs_multiply(res, lbfgs_data, x, α, β)
  return LBFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, false, lbfgs_data)
end

LBFGSOperator(n::I; kwargs...) where {I <: Integer} = LBFGSOperator(Float64, n; kwargs...)

function push_common!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
  ys::T,
) where {T, I, F1, F2, F3}
  # op.counters.updates += 1
  data = op.data
  insert = data.insert

  data.s[insert] .= s
  data.y[insert] .= y
  data.ys[insert] = ys
  if op.data.scaling
    !iszero(data.scaling_factor) && (data.opnorm_upper_bound -= 1 / op.data.scaling_factor)
    op.data.scaling_factor = ys / dot(y, y)
    !iszero(data.scaling_factor) && (data.opnorm_upper_bound += 1 / op.data.scaling_factor)
  end

  # Update arrays a and b used in forward products.
  if !op.inverse
    data.opnorm_upper_bound -= data.norm_b[insert]^2
    data.b[insert] .= y ./ sqrt(ys)
    data.norm_b[insert] = norm(data.b[insert])
    data.opnorm_upper_bound += data.norm_b[insert]^2

    @inbounds for i = 1:(data.mem)
      k = mod(insert + i - 1, data.mem) + 1
      if data.ys[k] != 0
        data.a[k] .= data.s[k] ./ data.scaling_factor  # B₀ = I / γ.

        @inbounds for j = 1:(i - 1)
          l = mod(insert + j - 1, data.mem) + 1
          if data.ys[l] != 0
            data.a[k] .+= dot(data.b[l], data.s[k]) .* data.b[l]
            data.a[k] .-= dot(data.a[l], data.s[k]) .* data.a[l]
          end
        end
        data.a[k] ./= sqrt(dot(data.s[k], data.a[k]))
      end
    end
  end

  op.data.insert = mod(insert, data.mem) + 1
  return op
end

"""
    push!(op, s, y)
    push!(op, s, y, Bs)
    push!(op, s, y, α, g)
    push!(op, s, y, α, g, Bs)

Push a new {s,y} pair into a L-BFGS operator.
The second calling sequence is used for forward updating damping, using the preallocated vector `Bs`.
If the operator is damped, the first call will create `Bs` and call the second call.
The third and fourth calling sequences are used in inverse LBFGS updating in conjunction with damping,
where α is the most recent steplength and g the gradient used when solving `d=-Hg`.
"""
function push!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
) where {T, I, F1, F2, F3}
  if op.data.damped
    return push!(op, s, y, similar(s))
  end
  ys = dot(y, s)
  σ₂ = op.data.σ₂
  σ₃ = op.data.σ₃

  if ys <= eps(eltype(op))
    # op.counters.rejects +=1
    return op
  end

  push_common!(op, s, y, ys)
end

function push!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
  Bs::Vector{T},
) where {T, I, F1, F2, F3}
  if !op.data.damped
    error("This push! should be used for damped operators")
  elseif op.inverse
    error("This function be used for forward operators. Use push!(op, s, y, α, g, Bs) instead.")
  end
  ys = dot(y, s)
  σ₂ = op.data.σ₂
  σ₃ = op.data.σ₃

  # Powell's damped update strategy
  mul!(Bs, op, s, one(T), zero(T))
  sBs = dot(s, Bs)
  damp = false
  if ys < (1 - σ₂) * sBs
    θ = σ₂ * sBs / (sBs - ys)
    damp = true
  elseif ys > (1 + σ₃) * sBs
    θ = σ₃ * sBs / (ys - sBs)
    damp = true
  end
  if damp
    y = θ .* y .+ (1 - θ) .* Bs  # damped y
    ys = θ * ys + (1 - θ) * sBs
  end

  push_common!(op, s, y, ys)
end

function push!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
  α::T,
  g::Vector{T},
  Bs::Vector{T},
) where {T, I, F1, F2, F3}
  if !op.data.damped
    error("This push! should be used for damped operators")
  elseif !op.inverse
    error("This function be used for inverse operators. Use push!(op, s, y, Bs) instead.")
  end
  ys = dot(y, s)
  σ₂ = op.data.σ₂
  σ₃ = op.data.σ₃

  # Powell's damped update strategy
  Bs .= -α .* g
  sBs = dot(s, Bs)
  damp = false
  if ys < (1 - σ₂) * sBs
    θ = σ₂ * sBs / (sBs - ys)
    damp = true
  elseif ys > (1 + σ₃) * sBs
    θ = σ₃ * sBs / (ys - sBs)
    damp = true
  end
  if damp
    y .= θ .* y .+ (1 - θ) .* Bs  # damped y
    ys = θ * ys + (1 - θ) * sBs
  end

  push_common!(op, s, y, ys)
end

function push!(
  op::LBFGSOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
  α::T,
  g::Vector{T},
) where {T, I, F1, F2, F3}
  push!(op, s, y, α, g, similar(g))
end

"""
    diag(op)
    diag!(op, d)
Extract the diagonal of a L-BFGS operator in forward mode.
"""
function diag(op::LBFGSOperator{T}) where {T}
  d = Vector{T}(undef, op.nrow)
  diag!(op, d)
end

function diag!(op::LBFGSOperator{T}, d) where {T}
  op.inverse && throw(
    LinearOperatorException("only the diagonal of a forward L-BFGS approximation is available"),
  )
  data = op.data

  fill!(d, 1)
  data.scaling && (d ./= data.scaling_factor)

  @inbounds for i = 1:(data.mem)
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0
      d .+= data.b[k] .^ 2 .- data.a[k] .^ 2
    end
  end
  return d
end

"""
    reset!(data)
Resets the given LBFGS data.
"""
function reset!(data::LBFGSData{T, I}, inverse::Bool) where {T, I <: Integer}
  for i = 1:(data.mem)
    fill!(data.s[i], 0)
    fill!(data.y[i], 0)
    if !inverse
      fill!(data.a[i], 0)
      fill!(data.b[i], 0)
    end
  end
  fill!(data.ys, 0)
  fill!(data.α, 0)
  data.scaling_factor = T(1)
  data.insert = 1
  return data
end

"""
    reset!(op)
Resets the LBFGS data of the given operator.
"""
function reset!(op::LBFGSOperator)
  reset!(op.data, op.inverse)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
