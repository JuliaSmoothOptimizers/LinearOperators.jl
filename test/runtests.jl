using Arpack, Test, TestSetExtensions, LinearOperators
using LinearAlgebra, LDLFactorizations, SparseArrays, JLArrays
using Zygote
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
include("gpu/test_S_kwarg.jl")
include("gpu/jlarrays.jl")
if Sys.isapple() && occursin("arm64", Sys.MACHINE)
    using Metal
    include("gpu/metal.jl")
end
