module LinearOperatorsOpNormExt

using LinearOperators
using Arpack
using TSVD
using LinearAlgebra
using GenericLinearAlgebra
import Arpack: eigs, svds

# Import the function from the main package to add methods to it
import LinearOperators: estimate_opnorm

function estimate_opnorm(B; kwargs...)
  _estimate_opnorm(B, eltype(B); kwargs...)
end

function _estimate_opnorm(B, ::Type{T}; kwargs...) where {T}
  _, s, _ = tsvd(B, 1)
  return s[1], true
end

function _estimate_opnorm(
  B::Union{Hermitian, Symmetric{<:Real}},
  ::Type{T};
  kwargs...,
) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
  return opnorm_eig(B; kwargs...)
end

function _estimate_opnorm(
  B,
  ::Type{T};
  kwargs...,
) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
  if ishermitian(B)
    return opnorm_eig(B; kwargs...)
  else
    return opnorm_svd(B; kwargs...)
  end
end

function opnorm_eig(B; max_attempts::Int = 3, tiny_dense_threshold = 5)
  n = size(B, 1)

  if n ≤ tiny_dense_threshold
    return maximum(abs, eigen(Matrix(B)).values), true
  end

  nev = 1
  ncv = max(20, 2*nev + 1)

  for attempt = 1:max_attempts
    try
      d, nconv, _, _, _ = eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)

      if nconv == 1
        return abs(d[1]), true
      end

    catch e
      if e isa Arpack.ARPACKException ||
         occursin("ARPACK", string(e)) ||
         occursin("AUPD", string(e))
        if ncv >= n
          @warn "ARPACK failed and NCV cannot be increased further." exception=e
          rethrow(e)
        end
      else
        rethrow(e)
      end
    end

    if attempt < max_attempts
      old_ncv = ncv
      ncv = min(2 * ncv, n)
      if ncv > old_ncv
        @warn "opnorm_eig: increasing NCV from $old_ncv to $ncv and retrying."
      else
        break
      end
    end
  end

  return NaN, false
end

function opnorm_svd(B; max_attempts::Int = 3, tiny_dense_threshold = 5)
  m, n = size(B)

  if min(m, n) ≤ tiny_dense_threshold
    return maximum(svd(Matrix(B)).S), true
  end

  min_dim = min(m, n)

  nsv = 1
  ncv = 10

  for attempt = 1:max_attempts
    try
      s, nconv, _, _, _ = svds(B; nsv = nsv, ncv = ncv, ritzvec = false, check = 1)

      if nconv >= 1
        return maximum(s.S), true
      end

    catch e
      if e isa Arpack.ARPACKException ||
         occursin("ARPACK", string(e)) ||
         occursin("AUPD", string(e))
        if ncv >= min_dim
          @warn "ARPACK failed and NCV cannot be increased further." exception=e
          rethrow(e)
        end
      else
        rethrow(e)
      end
    end

    if attempt < max_attempts
      old_ncv = ncv
      ncv = min(2 * ncv, min_dim)
      if ncv > old_ncv
        @warn "opnorm_svd: increasing NCV from $old_ncv to $ncv and retrying."
      else
        break
      end
    end
  end

  return NaN, false
end

end
