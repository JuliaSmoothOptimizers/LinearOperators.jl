export LSR1Operator, diag  #, InverseLSR1Operator

"A data type to hold information relative to LSR1 operators."
mutable struct LSR1Data{T}
  mem :: Int
  scaling :: Bool
  scaling_factor :: T
  s   :: Matrix{T}
  y   :: Matrix{T}
  ys  :: Vector{T}
  a   :: Matrix{T}
  as  :: Vector{T}
  insert :: Int
end

function LSR1Data(T :: DataType, n :: Int; mem :: Int=5, scaling :: Bool=true, inverse :: Bool=false)
  inverse && @warn "inverse LSR1 operator not yet implemented"
  LSR1Data{T}(max(mem, 1),
              scaling,
              convert(T, 1),
              zeros(T, n, mem),
              zeros(T, n, mem),
              zeros(T, mem),
              zeros(T, n, mem),
              zeros(T, mem),
              1)
end

LSR1Data(n :: Int; kwargs...) = LSR1Data(Float64, n; kwargs...)

"A type for limited-memory SR1 approximations."
mutable struct LSR1Operator{T} <: AbstractLinearOperator{T}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod     # apply the operator to a vector
  tprod    # apply the transpose operator to a vector
  ctprod   # apply the transpose conjugate operator to a vector
  inverse :: Bool
  data :: LSR1Data{T}
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
end

LSR1Operator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod, inverse::Bool, data::LSR1Data{T}) where T =
  LSR1Operator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, inverse, data, 0, 0, 0)

"""
    LSR1Operator(T, n; [mem=5, scaling=false)
    LSR1Operator(n; [mem=5, scaling=false)

Construct a limited-memory SR1 approximation in forward form. If the type `T` is
omitted, then `Float64` is used.
"""
function LSR1Operator(T :: DataType, n :: Int; kwargs...)
  lsr1_data = LSR1Data(T, n; kwargs...)

  function lsr1_multiply(data :: LSR1Data, x :: AbstractArray)
    # Multiply operator with a vector.

    result_type = promote_type(T, eltype(x))
    q = convert(Array{result_type}, copy(x))

    data.scaling && (q ./= data.scaling_factor)  # q = B₀ * x

    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        @views q .+= dot(data.a[:, k], x) / data.as[k] * data.a[:, k]
      end
    end
    return q
  end

  prod = @closure x -> lsr1_multiply(lsr1_data, x)
  return LSR1Operator{T}(n, n, true, true,
                         prod,
                         nothing, nothing,
                         false,
                         lsr1_data)
end

LSR1Operator(n :: Int; kwargs...) = LSR1Operator(Float64, n; kwargs...)


"""
    push!(op, s, y)

Push a new {s,y} pair into a L-SR1 operator.
"""
function push!(op :: LSR1Operator, s :: AbstractVector, y :: AbstractVector)

  # op.counters.updates += 1
  data = op.data
  Bs = op * s
  ymBs = y - Bs
  ys = dot(y, s)

  well_defined = abs(dot(ymBs, s)) ≥ 1.0e-8 + 1.0e-8 * norm(ymBs)^2

  sufficient_curvature = true
  scaling_condition = true
  if data.scaling
    sufficient_curvature = abs(ys) ≥ 1.0e-8
    if sufficient_curvature
      scaling_factor = ys / dot(y, y)
      scaling_condition = norm(y - s / scaling_factor) >= 1.0e-8
    end
  end

  if !(well_defined && sufficient_curvature && scaling_condition)
    @debug "LSR1 update rejected" well_defined sufficient_curvature scaling_condition
    # op.counters.rejects += 1
    return op
  end

  data.s[:, data.insert] .= s
  data.y[:, data.insert] .= y
  data.ys[data.insert] = ys

  # update scaling factor
  data.scaling && (data.scaling_factor = ys / dot(y, y))

  # update next insertion position
  data.insert = mod(data.insert, data.mem) + 1

  # update rank-1 terms
  for i = 1 : data.mem
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0
      data.a[:, k] .= data.y[:, k] - data.s[:, k] / data.scaling_factor  # = y - B₀ * s
      for j = 1 : i-1
        l = mod(data.insert + j - 2, data.mem) + 1
        if data.ys[l] != 0
          @views data.a[:, k] .-= dot(data.a[:, l], data.s[:, k]) / data.as[l] * data.a[:, l]
        end
      end
      @views data.as[k] = dot(data.a[:, k], data.s[:, k])
    end
  end

  return op
end


"""
    diag(op)

Extract the diagonal of a L-SR1 operator in forward mode.
"""
function diag(op :: LSR1Operator{T}) where T
  op.inverse && throw("only the diagonal of a forward L-SR1 approximation is available")
  data = op.data

  d = ones(T, op.nrow)
  data.scaling && (d ./= data.scaling_factor)

  for i = 1 : data.mem
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0.0
      for j = 1 : op.nrow
        d[j] += data.a[j, k]^2 / data.as[k]
      end
    end
  end
  return d
end

"""
    reset!(data)

Reset the given LSR1 data.
"""
function reset!(data :: LSR1Data{T}) where T
  fill!(data.s, 0)
  fill!(data.y, 0)
  fill!(data.ys, 0)
  fill!(data.a, 0)
  fill!(data.as, 0)
  data.scaling_factor = T(1)
  data.insert = 1
  return data
end

"""
    reset!(op)

Resets the LSR1 data of the given operator.
"""
function reset!(op :: LSR1Operator)
  reset!(op.data)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
