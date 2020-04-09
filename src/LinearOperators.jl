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

# Special operators
include("linalg.jl")
include("PreallocatedLinearOperators.jl")
include("special-operators.jl")
include("TimedOperators.jl")

# Utilities
include("utilities.jl")

include("deprecated.jl")

end # module
