export LSR1Operator, reset!  #, InverseLSR1Operator

"A data type to hold information relative to LSR1 operators."
type LSR1Data
  mem :: Int
  scaling :: Bool
  scaling_factor :: Float64
  s   :: Array
  y   :: Array
  ys  :: Vector
  a   :: Array
  as  :: Array
  insert :: Int

  function LSR1Data(n :: Int, mem :: Int;
                     dtype :: DataType=Float64, scaling :: Bool=false, inverse :: Bool=true)
    return new(max(mem, 1),
               scaling,
               1.0,
               zeros(dtype, n, mem),
               zeros(dtype, n, mem),
               zeros(dtype, mem),
               zeros(dtype, n, mem),
               zeros(dtype, mem),
               1)
  end
end

"A type for limited-memory SR1 approximations."
type LSR1Operator <: AbstractLinearOperator
  nrow   :: Int
  ncol   :: Int
  dtype   :: DataType
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
  inverse :: Bool
  data :: LSR1Data
end

"Construct a limited-memory SR1 approximation in forward form."
function LSR1Operator(n, mem :: Int=5; dtype :: DataType=Float64, scaling :: Bool=false)
  lsr1_data = LSR1Data(n, mem, dtype=dtype, scaling=scaling, inverse=false)

  function lsr1_multiply(data :: LSR1Data, x :: Array)
    # Multiply operator with a vector.

    if dtype == typeof(x[1])
      q = copy(x)
    else
      result_type = promote_type(dtype, typeof(x[1]))
      q = convert(Array{result_type}, x)
    end

    data.scaling && (q[:] /= data.scaling_factor)

    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        q[:] += dot(data.a[:, k], x) / data.as[k] * data.a[:, k]
      end
    end
    return q
  end

  return LSR1Operator(n, n, dtype, true, true,
                      x -> lsr1_multiply(lsr1_data, x),
                      Nullable{Function}(),
                      Nullable{Function}(),
                      false,
                      lsr1_data)
end


"Push a new {s,y} pair into a L-SR1 operator."
function push!(op :: LSR1Operator, s :: Vector, y :: Vector)

  # op.counters.updates += 1
  data = op.data
  Bs = op * s
  ymBs = y - Bs
  ys = dot(y, s)

  well_defined = abs(dot(ymBs, s)) >= 1.0e-8 + 1.0e-8 * norm(s) * norm(ymBs)

  sufficient_curvature = true
  scaling_condition = true
  y_neq_s = true
  if data.scaling
    sufficient_curvature = abs(ys) >= 1.0e-8
    if sufficient_curvature
      scaling_factor = ys / dot(y, y)
      scaling_condition = norm(y - s / scaling_factor) >= 1.0e-8
    end
  end

  if ~(well_defined && sufficient_curvature && scaling_condition && y_neq_s)
    # op.counters.rejects += 1
    return op
  end

  data.s[:, data.insert] = s
  data.y[:, data.insert] = y
  data.ys[data.insert] = ys

  # update scaling factor
  data.scaling && (data.scaling_factor = ys / dot(y, y))

  # update next insertion position
  data.insert = mod(data.insert, data.mem) + 1

  # update rank-1 terms
  for i = 1 : data.mem
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0.0
      data.a[:, k] = data.y[:, k] - data.s[:, k] / data.scaling_factor
      for j = 1 : i-1
        l = mod(data.insert + j - 2, data.mem) + 1
        if data.ys[l] != 0.0
          data.a[:, k] -= dot(data.a[:, l], data.s[:, k]) / data.as[l] * data.a[:, l]
        end
      end
      data.as[k] = dot(data.a[:, k], data.s[:, k])
    end
  end

  return op
end


"Extract the diagonal of a L-SR1 operator in forward mode."
function diag(op :: LSR1Operator)
  op.inverse && throw("only the diagonal of a forward L-SR1 approximation is available")
  data = op.data

  d = ones(op.nrow)
  data.scaling && (d[:] /= data.scaling_factor)

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

function reset!(data :: LSR1Data)
  data.s[:] = 0
  data.y[:] = 0
  data.ys[:] = 0
  data.a[:] = 0
  data.as[:] = 0
  data.insert = 1
end

function reset!(op :: LSR1Operator)
  reset!(op.data)
end
