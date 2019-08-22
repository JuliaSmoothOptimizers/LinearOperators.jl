# L-BFGS operator using the compact L-BFGS formula
#
# The formula is described in
#
# Byrd RH, Nocedal J, Schnabel RB. Representations of quasi-Newton matrices
# and their use in limited memory methods.
# Mathematical Programming. 1994 Jan 1;63(1-3):129-56.
#
# The operator has the form
#
#       B = B₀ - W M Wᵀ
#
# where B₀ = θI, W = [Y, θS], and M = [-D   Lᵀ ]⁻¹
#                                     [ L  θSᵀS]
#
# See the article for more details

export CompactLBFGSOperator, reset!, diag

################################################################################
# L-BFGS data definition                                                       #
################################################################################

"A type to hold the information relative to the Compact LBFGS operator"
mutable struct CompactLBFGSData{T}
  mem :: Int                       # Memory
  θ :: T                           # Scaling factor
  ws :: Vector{Vector{T}}          # S storage
  wy :: Vector{Vector{T}}          # Y storage
  sy :: SubArray{T}                # SᵀY storage (lower triangle only)
  ss :: SubArray{T}                # SᵗS storage (upper triangle only)
  wt :: UpperTriangular{T}         # Cholesky factorization of (θSᵀS + LD⁻¹Lᵀ)
  head :: Int                      # Index of the oldest pair
  col :: Int                       # Number of stored pairs
end

function CompactLBFGSData(T :: DataType, n :: Int, mem :: Int)
  # Storage for wt
  wt = view(zeros(T, mem, mem), Int[], Int[])
  # Storage for ss and sy
  sssy = zeros(T, mem, mem + 1)
  CompactLBFGSData{T}(max(mem, 1),                    # mem
                      one(T),                         # θ
                      [zeros(T, n) for i = 1 : mem],  # ws
                      [zeros(T, n) for i = 1 : mem],  # wy
                      view(sssy, :, 1 : mem),         # sy
                      view(sssy, :, 2 : mem + 1),     # ss
                      UpperTriangular(wt),            # wt
                      1,                              # head
                      0)                              # col
end

CompactLBFGSData(n :: Int, mem :: Int) = CompactLBFGSData(Float64, n, mem)

################################################################################
# L-BFGS operations                                                            #
################################################################################

"""
    wtrtimes!(q, data, v)

Performs the product q = Wᵀ * v, where v is a n-vector and q is a vector of size
at least 2 * data.col
"""
function wtrtimes!(q :: AbstractVector, data :: CompactLBFGSData, v :: AbstractVector)
  # [ q₁ ] = [  Yᵀ * v ]
  # [ q₂ ] = [ θSᵀ * v ]
  k = data.head
  for i = 1 : data.col
    q[i] = dot(data.wy[k], v)
    q[i+data.col] = data.θ * dot(data.ws[k], v)
    k = mod(k, data.mem) + 1
  end
end

"""
    wtimes!(v, data, p)

Performs the product v = W * p, where v is a n-vector and p is a vector of size
at least 2 * data.col
"""
function wtimes!(v :: AbstractVector, data :: CompactLBFGSData, p :: AbstractVector)
  # v = [ Y   θS ] * [ p₁ ]
  #                  [ p₂ ]
  fill!(v, 0)
  k = data.head
  for i = 1 : data.col
    @. v += p[i] * data.wy[k] + data.θ * p[i+data.col] * data.ws[k]
    k = mod(k, data.mem) + 1
  end
end

"""
    mtimes!(p, data, q)

Computes the product p = M * q, where p and q are vectors of size
at least 2 * data.col
"""
function mtimes!(p :: AbstractVector, data :: CompactLBFGSData{T}, q :: AbstractVector) where T
  # Compute  [ p₁ ]  = [ -D^½  D^(-½)*Lᵀ ]⁻¹ * [  D^½    0 ]⁻¹ * [ q₁ ]
  #          [ p₂ ]    [ 0         Jᵀ    ]     [ -LD^-½  J ]     [ q₂ ]

  data.col == 0 && return

  # First solve [  D^½    0 ] [ p₁ ] = [ q₁ ]
  #             [ -LD^-½  J ] [ p₂ ]   [ q₂ ]

  # Solve J * p₂ = q₂ + L * D⁻¹ q₁
  p[data.col+1] = q[data.col+1]
  for i = 2 : data.col
    i2 = data.col + i
    s = zero(T)
    for k = 1 : i - 1
      s += data.sy[i,k] * q[k] / data.sy[k,k]
    end
    p[i2] = q[i2] + s
  end
  p₂ = view(p, 1 + data.col : 2 * data.col)
  # Solve the triangular system
  p₂ .= data.wt' \ p₂

  # Solve D^½ * p₁ = q₁
  for i = 1 : data.col
    p[i] = q[i] / sqrt(data.sy[i,i])
  end

  # Second solve [ -D^½  D^(-½)*Lᵀ ] [ p₁ ] = [ p₁ ]
  #              [ 0         Jᵀ    ] [ p₂ ] = [ p₂ ]

  # Solve Jᵀ * p₂ = p₂
  p₂ .= data.wt \ p₂

  # Compute p₁ = -D^(-½) * (p₁ - D^(-½) * Lᵀ * p₂)
  #            = -D^(-½) * p₁ + D⁻¹ * Lᵀ * p₂.
  for i = 1 : data.col
    p[i] = -p[i] / sqrt(data.sy[i,i])
    s = zero(T)
    for k = i + 1 : data.col
      s = s + data.sy[k,i] * p[data.col+k] / data.sy[i,i]
    end
    p[i] += s
  end
end

"""
    push!(data, s, y)

Push a new {s,y} pair into the compact L-BFGS storage
"""
function push!(data :: CompactLBFGSData{T}, s :: Vector, y :: Vector) where T
  ys = dot(y, s)
  ys <= 1.0e-20 && return
  # Set the column index where to put the new pair
  if data.col < data.mem
    data.col += 1
    insert = data.col
    # If the memory is not full, wt is of type UpperTriangular{SubArray{Array}}
    wt = data.wt.data.parent
    # Enlarge the view on wt
    if data.col < data.mem
      data.wt = UpperTriangular(view(wt, 1 : data.col, 1 : data.col))
    else
      data.wt = UpperTriangular(wt)
    end
  else
    insert = data.head
    data.head = mod(data.head, data.mem) + 1
    #If the memory is full, wt is of type UpperTriangular{Array}
    wt = data.wt.data

    # Move old information in sy and ss
    for j = 1 : data.col - 1
      data.ss[1:j,j] = data.ss[2:j+1,j+1]
      data.sy[j:data.col-1,j] = data.sy[j+1:data.col,j+1]
    end
  end

  # Update S, Y, and θ
  copyto!(data.ws[insert], s)
  copyto!(data.wy[insert], y)
  data.θ = dot(y, y) / ys

  # Add new information in sy and ss
  k = data.head
  for j = 1 : data.col - 1
    data.sy[data.col,j] = dot(s, data.wy[k])
    data.ss[j,data.col] = dot(s, data.ws[k])
    k = mod(k, data.mem) + 1
  end
  data.ss[data.col,data.col] = dot(s, s)
  data.sy[data.col,data.col] = ys

  # Prepare Jᵀ (to be stored in data.wt)
  # wt is the actual array containing the components of data.wt

  # Form the upper half of θ * ss + LD⁻¹Lᵀ,
  # Store it in the upper triangle of wt[1:col, 1:col]
  for i = 1:data.col
    wt[1,i] = data.θ * data.ss[1,i]
  end
  for i = 2 : data.col
    for j = i : data.col
      k1 = min(i, j) - 1
      tmp = zero(T)
      for k = 1 : k1
        tmp += data.sy[i,k] * data.sy[j,k] / data.sy[k,k]
      end
      wt[i,j] = tmp + data.θ * data.ss[i,j]
    end
  end
  cholesky!(Symmetric(data.wt.data))
end

"""
    reset!(data)

Resets the given compact LBFGS data.
"""
function reset!(data :: CompactLBFGSData{T}) where T
  data.θ = one(T)
  for k = 1 : data.mem
    fill!(data.ws[k], zero(T))
    fill!(data.wy[k], zero(T))
  end
  fill!(data.sy, zero(T))
  fill!(data.ss, zero(T))
  if data.col < data.mem
    wt = data.wt.data.parent
  else
    wt = data.wt.data
  end
  fill!(wt, zero(T))
  data.wt = UpperTriangular(view(wt, Int[], Int[]))
  data.head = 1
  data.col = 0
  return data
end

################################################################################
# Operator definition                                                          #
################################################################################

"A type for a compact L-BFGS approximation"
mutable struct CompactLBFGSOperator{T,F1<:FuncOrNothing,F2<:FuncOrNothing,
                        F3<:FuncOrNothing} <: AbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1  # apply the operator to a vector
  tprod  :: F2  # apply the transpose operator to a vector
  ctprod :: F3  # apply the transpose conjugate operator to a vector
  data :: CompactLBFGSData{T}
end

function CompactLBFGSOperator(T :: DataType, n :: Int, mem :: Int=5)
  lbfgs_data = CompactLBFGSData(T, n, mem)

  function lbfgs_multiply(data :: CompactLBFGSData{T}, x :: AbstractVector) where T
    # Multiply operator with a vector
    # B * v = θ * v - W M Wᵀ v

    res_type = promote_type(T, eltype(x))
    v = similar(x, res_type)

    p = zeros(res_type, 2 * data.col)
    q = zeros(res_type, 2 * data.col)

    wtrtimes!(q, data, x) # q = Wᵀ * v
    mtimes!(p, data, q)   # p = M  * q
    wtimes!(v, data, p)   # v = W  * p
    @. v = data.θ * x - v # v = θ * x - v

    return v
  end

  prod = @closure x -> lbfgs_multiply(lbfgs_data, x)
  F = typeof(prod)
  return CompactLBFGSOperator{T,F,F,F}(n, n, true, true, prod, prod, prod, lbfgs_data)
end

CompactLBFGSOperator(n :: Int, mem :: Int=5) = CompactLBFGSOperator(Float64, n, mem)

"""
    diag(op)

Extract the diagonal of a L-BFGS operator in forward mode
"""
function diag(op :: CompactLBFGSOperator{T}) where T
  d = op.data.θ * ones(T, op.nrow)
  op.data.col == 0 && return d

  #
  # Compute the diagonal of W M Wᵀ
  # where M = K⁻ᵀ * N *  K⁻¹ and N = [ -I  0 ]
  #                                  [  0  I ]
  #
  col  = op.data.col
  head = op.data.head
  p₂ = zeros(T, col)
  for n = 1 : op.nrow
    #
    # We compute for each n
    #
    #       dₙ = θ - ⟨eₙ, WMWᵀeₙ⟩
    #          = θ - ⟨K⁻¹Wᵀeₙ, NK⁻¹Wᵀeₙ⟩
    #          = θ - ⟨K⁻¹q, NK⁻¹q⟩
    #
    # with K = [  D^½    0 ] and q = Wᵀ eₙ = [     Y[n,:] ] = [ q₁ ]
    #          [ -LD^-½  J ]                 [ θ * S[n,:] ]   [ q₂ ]
    #

    # Step 1: Solve J * p₂ = q₂ + L * D⁻¹ q₁
    p₂[1] = op.data.θ * op.data.ws[head][n]
    k = mod(head, op.data.mem) + 1
    for i = 2 : col
      i2 = col + i
      s = zero(T)
      k2 = head
      for j = 1 : i - 1
        s += op.data.sy[i,j] * op.data.wy[k2][n] / op.data.sy[j,j]
        k2 = mod(k2, op.data.mem) + 1
      end
      p₂[i] = op.data.θ * op.data.ws[k][n] + s
      k = mod(k, op.data.mem) + 1
    end
    # Solve the triangular system
    p₂ .= op.data.wt' \ p₂

    # Step 2: Solve D^½ * p₁ = q₁
    #         Compute dₙ = θ + ‖p₁‖² - ‖p₂‖²
    k = head
    for i = 1 : col
      p₁i² = op.data.wy[k][n]^2 / op.data.sy[i,i]
      d[n] += p₁i² - p₂[i]^2
      k = mod(k, op.data.mem) + 1
    end
  end
  return d
end

"""
    push!(op, s, y)

Push a new {s,y} pair into a compact L-BFGS operator
"""
function push!(op :: CompactLBFGSOperator, s :: Vector, y :: Vector)
  push!(op.data, s, y)
  return op
end

"""
    reset!(op)

Reset the given compact L-BFGS operator
"""
function reset!(op :: CompactLBFGSOperator)
  reset!(op.data)
  return op
end
