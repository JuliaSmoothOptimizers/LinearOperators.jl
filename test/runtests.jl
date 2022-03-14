using Arpack, Test, TestSetExtensions, LinearOperators
using LinearAlgebra, SparseArrays
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
