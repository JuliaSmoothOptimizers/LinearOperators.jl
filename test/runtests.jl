using Arpack, Test, TestSetExtensions, LinearOperators
using LinearAlgebra, LDLFactorizations, SparseArrays, JLArrays
using Zygote
if Sys.isapple() && occursin("arm64", Sys.MACHINE)
    using Metal
end
include("test_aux.jl")

include("test_linop.jl")
include("test_linop_allocs.jl")
include("test_adjtrans.jl")
include("test_cat.jl")
include("test_lbfgs.jl")
include("test_lsr1.jl")
include("test_kron.jl")
include("test_callable.jl")
include("test_deprecated.jl")
include("test_normest.jl")
include("test_diag.jl")
include("test_chainrules.jl")
include("test_solve_shifted_system.jl")
include("test_S_kwarg.jl")
