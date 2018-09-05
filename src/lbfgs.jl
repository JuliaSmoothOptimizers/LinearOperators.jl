export LBFGSOperator, InverseLBFGSOperator, reset!


"A data type to hold information relative to LBFGS operators."
mutable struct LBFGSData{T}
  mem :: Int
  scaling :: Bool
  scaling_factor :: T
  damped :: Bool
  damp_factor :: T
  s   :: Array{T}
  y   :: Array{T}
  ys  :: Vector{T}
  α   :: Vector{T}
  a   :: Array{T}
  b   :: Array{T}
  insert :: Int
end

function LBFGSData(T :: DataType, n :: Int, mem :: Int;
                   scaling :: Bool=true, damped :: Bool=false, inverse :: Bool=true)
  LBFGSData{T}(max(mem, 1),
               scaling,
               convert(T, 1),
               damped,
               convert(T, 0.2),
               zeros(T, n, mem),
               zeros(T, n, mem),
               zeros(T, mem),
               inverse ? zeros(T, mem) : T[],
               inverse ? T[] : zeros(T, n, mem),
               inverse ? T[] : zeros(T, n, mem),
               1)
end

LBFGSData(n :: Int, mem :: Int; kwargs...) = LBFGSData(Float64, n, mem; kwargs...)

"A type for limited-memory BFGS approximations."
mutable struct LBFGSOperator{T,F1<:FuncOrVoid,F2<:FuncOrVoid,F3<:FuncOrVoid} <: AbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1  # apply the operator to a vector
  tprod  :: F2  # apply the transpose operator to a vector
  ctprod :: F3  # apply the transpose conjugate operator to a vector
  inverse :: Bool
  data :: LBFGSData{T}
end


"""
    InverseLBFGSOperator(T, n, [mem=5; scaling=true])
    InverseLBFGSOperator(n, [mem=5; scaling=true])

Construct a limited-memory BFGS approximation in inverse form. If the type `T`
is omitted, then `Float64` is used.
"""
function InverseLBFGSOperator(T :: DataType, n :: Int, mem :: Int=5; kwargs...)

  kwargs = Dict(kwargs)
  delete!(kwargs, :inverse)
  lbfgs_data = LBFGSData(T, n, mem; inverse=true, kwargs...)

  function lbfgs_multiply(data :: LBFGSData, x :: AbstractArray)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.4, p. 178.

    result_type = promote_type(T, eltype(x))
    q = convert(Array{result_type}, copy(x))

    for i = 1 : data.mem
      k = mod(data.insert - i - 1, data.mem) + 1
      if data.ys[k] != 0
        data.α[k] = dot(data.s[:,k], q) / data.ys[k]
        q[:] -= data.α[k] * data.y[:,k]
      end
    end

    data.scaling && (q[:] *= data.scaling_factor)

    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        β = dot(data.y[:,k], q) / data.ys[k]
        q[:] += (data.α[k] - β) * data.s[:,k]
      end
    end

    return q
  end

  prod = @closure x -> lbfgs_multiply(lbfgs_data, x)
  F = typeof(prod)
  return LBFGSOperator{T,F,F,F}(n, n, true, true,
                                                 prod,
                                                 prod, prod,
                                                 true,
                                                 lbfgs_data)
end

InverseLBFGSOperator(n :: Int, mem :: Int=5; kwargs...) = InverseLBFGSOperator(Float64, n, mem; kwargs...)


"""
    LBFGSOperator(T, n, [mem=5; scaling=true])
    LBFGSOperator(n, [mem=5; scaling=true])

Construct a limited-memory BFGS approximation in forward form. If the type `T`
is omitted, then `Float64` is used.
"""
function LBFGSOperator(T :: DataType, n :: Int, mem :: Int=5; kwargs...)

  kwargs = Dict(kwargs)
  delete!(kwargs, :inverse)
  lbfgs_data = LBFGSData(T, n, mem; inverse=false, kwargs...)

  function lbfgs_multiply(data :: LBFGSData, x :: AbstractArray)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.6, p. 184.

    result_type = promote_type(T, eltype(x))
    q = convert(Array{result_type}, copy(x))

    data.scaling && (q[:] /= data.scaling_factor)

    # B = B₀ + Σᵢ (bᵢbᵢ' - aᵢaᵢ').
    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        q[:] += dot(data.b[:, k], x) * data.b[:, k] - dot(data.a[:, k], x) * data.a[:, k]
      end
    end
    return q
  end

  prod = @closure x -> lbfgs_multiply(lbfgs_data, x)
  F = typeof(prod)
  return LBFGSOperator{T,F,F,F}(n, n, true, true,
                                                 prod,
                                                 prod, prod,
                                                 false,
                                                 lbfgs_data)
end

LBFGSOperator(n :: Int, mem :: Int=5; kwargs...) = LBFGSOperator(Float64, n, mem; kwargs...)

"""
    push!(op, s, y)
    push!(op, s, y, α, g)

Push a new {s,y} pair into a L-BFGS operator.
The second calling sequence is used in inverse LBFGS updating in conjunction with damping,
where α is the most recent steplength and g the gradient used when solving `d=-Hg`.
In forward updating with damping, it is not necessary to supply α and g.
"""
function push!(op :: LBFGSOperator, s :: Vector, y :: Vector, α :: Real=1.0, g :: Vector=zero(y))

  ys = dot(y, s)
  σ₂ = 0.99
  σ₃ = 10.0

  if op.data.damped
    # Powell's damped update strategy
    Bs = op.inverse ? (- α * g) : (op * s)
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
      y = θ * y + (1 - θ) * Bs  # damped y
      ys = θ * ys + (1 - θ) * sBs
    end
  else
    if ys <= 1.0e-20
      # op.counters.rejects +=1
      return op
    end
  end

  # op.counters.updates += 1
  data = op.data
  insert = data.insert

  data.s[:, insert] = s
  data.y[:, insert] = y
  data.ys[insert] = ys

  op.data.scaling && (op.data.scaling_factor = ys / dot(y, y))

  # Update arrays a and b used in forward products.
  if !op.inverse
    data.b[:, insert] = y / sqrt(ys)

    for i = 1 : data.mem
      k = mod(insert + i - 1, data.mem) + 1
      if data.ys[k] != 0
        data.a[:, k] = data.s[:, k]   # B₀ = I.

        for j = 1 : i - 1
          l = mod(insert + j - 1, data.mem) + 1
          if data.ys[l] != 0
            data.a[:, k] += dot(data.b[:, l], data.s[:, k]) * data.b[:, l]
            data.a[:, k] -= dot(data.a[:, l], data.s[:, k]) * data.a[:, l]
          end
        end
        data.a[:, k] /= sqrt(dot(data.s[:, k], data.a[:, k]))
      end
    end
  end

  op.data.insert = mod(insert, data.mem) + 1
  return op
end


"""
    diag(op)

Extract the diagonal of a L-BFGS operator in forward mode.
"""
function diag(op :: LBFGSOperator{T}) where T
  op.inverse && throw(LinearOperatorException("only the diagonal of a forward L-BFGS approximation is available"))
  data = op.data

  d = ones(T, op.nrow)
  data.scaling && (d[:] /= data.scaling_factor)

  for i = 1 : data.mem
    k = mod(data.insert + i - 2, data.mem) + 1;
    if data.ys[k] != 0
      for j = 1 : op.nrow
        d[j] = d[j] + data.b[j, k]^2 - data.a[j, k]^2;
      end
    end
  end
  return d
end

"""
    reset!(data)

Resets the given LBFGS data.
"""
function reset!(data :: LBFGSData)
  fill!(data.s, 0)
  fill!(data.y, 0)
  fill!(data.ys, 0)
  fill!(data.α , 0)
  fill!(data.a, 0)
  fill!(data.b, 0)
  data.insert = 1
  return data
end

"""
    reset!(op)

Resets the LBFGS data of the given operator.
"""
function reset!(op :: LBFGSOperator)
  reset!(op.data)
  return op
end
