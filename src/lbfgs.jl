export LBFGSOperator, InverseLBFGSOperator


"A data type to hold information relative to LBFGS operators."
type LBFGSData
  mem :: Int
  scaling :: Bool
  scaling_factor :: Float64
  s   :: Array
  y   :: Array
  ys  :: Vector
  α   :: Vector
  a   :: Array
  b   :: Array
  insert :: Int

  function LBFGSData(n :: Int, mem :: Int;
                     dtype :: DataType=Float64, scaling :: Bool=false, inverse :: Bool=true)
    return new(max(mem, 1),
               scaling,
               1.0,
               zeros(dtype, n, mem),
               zeros(dtype, n, mem),
               zeros(dtype, mem),
               inverse ? zeros(dtype, mem) : dtype[],
               inverse ? dtype[] : zeros(dtype, n, mem),
               inverse ? dtype[] : zeros(dtype, n, mem),
               1)
  end
end


"A type for limited-memory BFGS approximations."
type LBFGSOperator <: AbstractLinearOperator
  nrow   :: Int
  ncol   :: Int
  dtype   :: DataType
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
  inverse :: Bool
  data :: LBFGSData
end


"Construct a limited-memory BFGS approximation in inverse form."
function InverseLBFGSOperator(n, mem :: Int=5; dtype :: DataType=Float64, scaling :: Bool=false)

  lbfgs_data = LBFGSData(n, mem, dtype=dtype, scaling=scaling)

  function lbfgs_multiply(data :: LBFGSData, x :: Array)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.4, p. 178.

    if dtype == typeof(x[1])
      q = copy(x)
    else
      result_type = promote_type(dtype, typeof(x[1]))
      q = convert(Array{result_type}, x)
    end

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

  return LBFGSOperator(n, n, dtype, true, true,
                       x -> lbfgs_multiply(lbfgs_data, x),
                       Nullable{Function}(),
                       Nullable{Function}(),
                       true,
                       lbfgs_data)
end


"Construct a limited-memory BFGS approximation in forward form."
function LBFGSOperator(n, mem :: Int=5; dtype :: DataType=Float64, scaling :: Bool=false)
  lbfgs_data = LBFGSData(n, mem, dtype=dtype, scaling=scaling, inverse=false)

  function lbfgs_multiply(data :: LBFGSData, x :: Array)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.6, p. 184.

    if dtype == typeof(x[1])
      q = copy(x)
    else
      result_type = promote_type(dtype, typeof(x[1]))
      q = convert(Array{result_type}, x)
    end

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

  return LBFGSOperator(n, n, dtype, true, true,
                       x -> lbfgs_multiply(lbfgs_data, x),
                       Nullable{Function}(),
                       Nullable{Function}(),
                       false,
                       lbfgs_data)
end


"Push a new {s,y} pair into a L-BFGS operator."
function push!(op :: LBFGSOperator, s :: Vector, y :: Vector)

  ys = dot(y, s)
  if ys <= 1.0e-20
    # op.counters.rejects +=1
    return op
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


"Extract the diagonal of a L-BFGS operator in forward mode."
function diag(op :: LBFGSOperator)
  op.inverse && throw("only the diagonal of a forward L-BFGS approximation is available")
  data = op.data

  d = ones(op.nrow)
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
