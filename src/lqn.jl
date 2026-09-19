export LQNOperator

"A data type to hold information relative to `LQNOperator`s."
mutable struct LQNData{T, I <: Integer}
  const mem::I
  const scaling::Bool
  scaling_factor::T
  opnorm_upper_bound::T # Upper bound for the operator norm ‖Bₖ‖₂ ≤ ‖B₀‖₂ + Σᵢ ‖aᵢ‖₂² + ‖bᵢ‖₂²
  const s::Vector{Vector{T}}
  const y::Vector{Vector{T}}
  const ys::Vector{T}
  const update_type::Vector{Symbol} # :empty, :bfgs or :sr1 for each memory slot
  const a::Vector{Vector{T}} # BFGS: -aaᵀ term; SR1: sign_a * aaᵀ term
  const b::Vector{Vector{T}} # +bbᵀ term, only used for BFGS slots
  const sign_a::Vector{T} # always -1 for BFGS slots, ±1 (sign of the SR1 denominator) for SR1 slots
  const norm_a::Vector{T}
  const norm_b::Vector{T}
  insert::I
  const Ax::Vector{T}
  const Bs::Vector{T}
  const tmp::Vector{T}
end

function LQNData(T::Type, n::I; mem::I = 5, scaling::Bool = true) where {I <: Integer}
  m = max(mem, I(1))
  LQNData{T, I}(
    m,
    scaling,
    convert(T, 1),
    convert(T, 1),
    [zeros(T, n) for _ = 1:m],
    [zeros(T, n) for _ = 1:m],
    zeros(T, m),
    fill(:empty, m),
    [zeros(T, n) for _ = 1:m],
    [zeros(T, n) for _ = 1:m],
    zeros(T, m),
    zeros(T, m),
    zeros(T, m),
    1,
    Vector{T}(undef, n),
    Vector{T}(undef, n),
    Vector{T}(undef, n),
  )
end

LQNData(n::I; kwargs...) where {I <: Integer} = LQNData(Float64, n; kwargs...)

"A type for a general limited-memory quasi-Newton approximation."
mutable struct LQNOperator{T, I <: Integer, F, Ft, Fct} <: AbstractQuasiNewtonOperator{T}
  const nrow::I
  const ncol::I
  const symmetric::Bool
  const hermitian::Bool
  const prod!::F    # apply the operator to a vector
  const tprod!::Ft    # apply the transpose operator to a vector
  const ctprod!::Fct   # apply the transpose conjugate operator to a vector
  const data::LQNData{T, I}
  nprod::I
  ntprod::I
  nctprod::I
end

LQNOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  data::LQNData{T, I},
) where {T, I <: Integer, F, Ft, Fct} = LQNOperator{T, I, F, Ft, Fct}(
  nrow,
  ncol,
  symmetric,
  hermitian,
  prod!,
  tprod!,
  ctprod!,
  data,
  0,
  0,
  0,
)

has_args5(op::LQNOperator) = true
isallocated5(op::LQNOperator) = true
storage_type(op::LQNOperator{T}) where {T} = Vector{T}

"""
    LQNOperator(T, n; [mem=5, scaling=true])
    LQNOperator(n; [mem=5, scaling=true])

Construct a limited-memory quasi-Newton approximation in forward form that, at each `push!`,
automatically chooses a BFGS or an SR1 update depending on which one satisfies its numerical
safeguards. If neither is well defined, the pair is rejected, exactly as in `LBFGSOperator` and
`LSR1Operator`.

A BFGS update is attempted first, since it preserves positive definiteness. If the curvature
condition `sᵀy > 0` or the well-definedness condition `sᵀBs > 0` fails, an SR1 update is
attempted instead, which can capture negative curvature at the cost of positive definiteness.
See [issue #257](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl/issues/257) for the
motivation behind this operator.

If the type `T` is omitted, then `Float64` is used.
"""
function LQNOperator(T::Type, n::I; kwargs...) where {I <: Integer}
  data = LQNData(T, n; kwargs...)

  function lqn_multiply(res::AbstractVector, data::LQNData, x::AbstractArray, α, β::T2) where {T2}
    # Multiply operator with a vector.
    # B = B₀ + Σᵢ (sign_a[i] * aᵢaᵢᵀ) + Σᵢ (bᵢbᵢᵀ for BFGS slots only).

    q = data.Ax
    q .= x ./ data.scaling_factor

    @inbounds for i = 1:(data.mem)
      k = mod(data.insert + i - 2, data.mem) + 1
      ut = data.update_type[k]
      if ut !== :empty
        ax = dot(data.a[k], x)
        q .+= (data.sign_a[k] * ax) .* data.a[k]
        if ut === :bfgs
          bx = dot(data.b[k], x)
          q .+= bx .* data.b[k]
        end
      end
    end
    if β == zero(T2)
      res .= α .* q
    else
      res .= α .* q .+ β .* res
    end
  end

  prod! = @closure (res, x, α, β) -> lqn_multiply(res, data, x, α, β)
  return LQNOperator{T}(n, n, true, true, prod!, prod!, prod!, data)
end

LQNOperator(n::I; kwargs...) where {I <: Integer} = LQNOperator(Float64, n; kwargs...)

"""
    push!(op, s, y)

Push a new {s,y} pair into a `LQNOperator`.
A BFGS update is used if it is well defined (`sᵀy > 0` and `sᵀBs > 0`); otherwise an SR1 update
is used if it is well defined; otherwise the pair is rejected and the operator is left unchanged.
"""
function push!(
  op::LQNOperator{T, I, F1, F2, F3},
  s::Vector{T},
  y::Vector{T},
) where {T, I, F1, F2, F3}
  data = op.data
  ϵ = eps(T)
  sNorm = norm(s)
  yNorm = norm(y)
  ys = dot(y, s)

  # Bs = B * s, computed with the operator as it stands before this update.
  Bs = data.Bs
  mul!(Bs, op, s, one(T), zero(T))
  sBs = dot(s, Bs)

  bfgs_ok = ys ≥ ϵ + ϵ * yNorm * sNorm && sBs ≥ ϵ + ϵ * norm(Bs) * sNorm

  update_type = :empty
  if bfgs_ok
    update_type = :bfgs
  else
    r = data.tmp
    r .= y .- Bs
    as = dot(s, r)
    sr1_ok = abs(as) ≥ ϵ + ϵ * norm(r) * sNorm
    sr1_ok && (update_type = :sr1)
  end

  if update_type === :empty
    # op.counters.rejects += 1
    @debug "LQN update rejected" bfgs_ok
    return op
  end

  insert = data.insert
  data.s[insert] .= s
  data.y[insert] .= y
  data.ys[insert] = ys
  data.update_type[insert] = update_type

  if data.scaling
    yy = dot(y, y)
    if yy > 0 && abs(ys) ≥ ϵ + ϵ * yNorm * sNorm
      data.scaling_factor = ys / yy
    end
  end

  data.insert = mod(insert, data.mem) + 1

  # Recompute the rank-one correction terms of every active slot, in chronological order, using
  # the current scaling factor. A slot whose safeguard no longer holds (e.g. because the scaling
  # factor changed, or an intervening SR1 update introduced negative curvature) is dropped.
  bound = one(T) / abs(data.scaling_factor)
  @inbounds for i = 1:(data.mem)
    k = mod(data.insert + i - 2, data.mem) + 1
    ut = data.update_type[k]
    ut === :empty && continue

    a = data.a[k]
    if ut === :bfgs
      a .= data.s[k] ./ data.scaling_factor # B₀ sₖ
    else
      a .= data.y[k] .- data.s[k] ./ data.scaling_factor # yₖ - B₀ sₖ
    end
    for j = 1:(i - 1)
      l = mod(data.insert + j - 2, data.mem) + 1
      lt = data.update_type[l]
      lt === :empty && continue
      c = data.sign_a[l] * dot(data.a[l], data.s[k])
      if ut === :bfgs
        a .+= c .* data.a[l]
        lt === :bfgs && (a .+= dot(data.b[l], data.s[k]) .* data.b[l])
      else
        a .-= c .* data.a[l]
        lt === :bfgs && (a .-= dot(data.b[l], data.s[k]) .* data.b[l])
      end
    end

    if ut === :bfgs
      sks = dot(data.s[k], a)
      if sks ≤ ϵ + ϵ * norm(a) * norm(data.s[k])
        data.update_type[k] = :empty # no longer well defined given the current history
        continue
      end
      a ./= sqrt(sks)
      data.sign_a[k] = -one(T)
      data.norm_a[k] = norm(a)

      b = data.b[k]
      b .= data.y[k] ./ sqrt(data.ys[k])
      data.norm_b[k] = norm(b)
      bound += data.norm_a[k]^2 + data.norm_b[k]^2
    else
      as = dot(data.s[k], a)
      if abs(as) ≤ ϵ + ϵ * norm(a) * norm(data.s[k])
        data.update_type[k] = :empty # no longer well defined given the current history
        continue
      end
      data.sign_a[k] = sign(as)
      a ./= sqrt(abs(as))
      data.norm_a[k] = norm(a)
      data.norm_b[k] = zero(T)
      bound += data.norm_a[k]^2
    end
  end
  data.opnorm_upper_bound = bound

  return op
end

"""
    diag(op)
    diag!(op, d)

Extract the diagonal of a `LQNOperator`.
"""
function diag(op::LQNOperator{T}) where {T}
  d = Vector{T}(undef, op.nrow)
  diag!(op, d)
end

function diag!(op::LQNOperator{T}, d) where {T}
  data = op.data

  fill!(d, 1)
  d ./= data.scaling_factor

  @inbounds for i = 1:(data.mem)
    k = mod(data.insert + i - 2, data.mem) + 1
    ut = data.update_type[k]
    if ut !== :empty
      d .+= data.sign_a[k] .* data.a[k] .^ 2
      ut === :bfgs && (d .+= data.b[k] .^ 2)
    end
  end
  return d
end

"""
    reset!(data)

Resets the given LQN data.
"""
function reset!(data::LQNData{T, I}) where {T, I <: Integer}
  for i = 1:(data.mem)
    fill!(data.s[i], 0)
    fill!(data.y[i], 0)
    fill!(data.a[i], 0)
    fill!(data.b[i], 0)
    data.update_type[i] = :empty
    data.sign_a[i] = zero(T)
    data.norm_a[i] = zero(T)
    data.norm_b[i] = zero(T)
  end
  fill!(data.ys, 0)
  data.scaling_factor = T(1)
  data.opnorm_upper_bound = T(1)
  data.insert = 1
  return data
end

"""
    reset!(op)

Resets the LQN data of the given operator.
"""
function reset!(op::LQNOperator)
  reset!(op.data)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
