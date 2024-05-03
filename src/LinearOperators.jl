module LinearOperators

using FastClosures, LinearAlgebra, Printf, SparseArrays

# Basic defitions
include("abstract.jl")
include("constructors.jl")

# Operations
include("operations.jl") # This first
include("adjtrans.jl")
include("cat.jl")
include("kron.jl")

# quasi-Newton operators
include("qn.jl")

# diagonal Hessian approximations
include("DiagonalHessianApproximation.jl")

# Special operators
include("linalg.jl")
include("special-operators.jl")
include("TimedOperators.jl")

# Utilities
include("utilities.jl")
include("deprecated.jl")

# lazy loading of chainrules for Julia < 1.9
@static if !isdefined(Base, :get_extension)
  import Requires
end

@static if !isdefined(Base, :get_extension)
  function __init__()
    Requires.@require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin
      include("../ext/LinearOperatorsChainRulesCoreExt.jl")
    end
    Requires.@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
      include("../ext/LinearOperatorsCUDAExt.jl")
    end
    Requires.@require LDLFactorizations = "40e66cde-538c-5869-a4ad-c39174c6795b" begin
      include("../ext/LinearOperatorsLDLFactorizationsExt.jl")
    end
  end
end

end # module
