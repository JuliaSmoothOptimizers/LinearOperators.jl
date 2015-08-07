# A [Julia](http://julialang.org) Linear Operator Package

[![Build Status](https://travis-ci.org/dpo/LinearOperators.jl.svg?branch=master)](https://travis-ci.org/dpo/LinearOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/l76yjaoqa4lyxhi7?svg=true)](https://ci.appveyor.com/project/dpo/linop-jl)
[![Coverage Status](https://coveralls.io/repos/dpo/LinearOperators.jl/badge.svg?branch=master)](https://coveralls.io/r/dpo/LinearOperators.jl?branch=master)

Operators behave like matrices but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector.

## Compatibility

Julia 0.3 and 0.4.

## How to Install

````JULIA
Pkg.clone("https://github.com/dpo/LinearOperators.jl.git")
````

## Example 1

Operators may be defined from matrices and combined using the usual operations, but the result is deferred until the operator is applied.

````JULIA
julia> A1 = rand(5,7);
julia> A2 = sprand(7,3,.3);
julia> op1 = LinearOperator(A1);
julia> op2 = LinearOperator(A2);
julia> op = op1 * op2;  # Does not form A1 * A2
julia> x = rand(3);
julia> y = op * x;
````

## Example 2

Operators may be defined to represent (approximate) inverses.

````JULIA
julia> A = rand(5,5); A = A' * A;
julia> op = opCholesky(A);  # Use, e.g., as a preconditioner
julia> v = rand(5);
julia> norm(A \ v - op * v) / norm(v)
1.6522645623951567e-14
````

## Example 3

Operators may be defined from functions. In the example below, the transposed isn't defined, but it may be inferred from the conjugate transposed. Missing operations are represented as [nullable](http://julia.readthedocs.org/en/latest/manual/types/?highlight=nullable#nullable-types-representing-missing-values) functions. Nullable types were introduced in Julia 0.4 but are provided in Julia 0.3 by [Compat.jl](https://github.com/JuliaLang/Compat.jl).

````JULIA
julia> using Compat  # only required if you use Julia 0.3.
julia> dft = LinearOperator(10, 10, Float64, false, false,
                            v -> fft(v),
                            Nullable{Function}(),  # this operation is "missing".
                            w -> ifft(w));
julia> x = rand(10);
julia> y = dft * x;
julia> norm(dft' * y - x)  # DFT is an orthogonal operator
2.2929868617541516e-16
julia> dft.' * y
julia> 10-element Array{Complex{Float64},1}:
  0.927514+0.227908im
 0.0472611+0.62094im
 ...
````

## Operators Available

Operator         | Description
-----------------|------------
`LinearOperator` | Base class. Useful to define operators from functions
`opEye`          | Identity operator
`opOnes`         | All ones operator
`opZeros`        | All zeros operator
`opDiagonal`     | Square (equivalent to `diagm()`) or rectangular diagonal operator
`opInverse`      | Equivalent to `\`
`opCholesky`     | More efficient than `opInverse` for symmetric positive definite matrices
`opHouseholder`  | Apply a Householder transformation `I-2hh'`
`opHermitian`    | Represent a symmetric/hermitian operator based on the diagonal and strict lower triangle

## Utility Functions

Function           | Description
-------------------|------------
`full`             | Convert an abstract operator to a dense array
`check_ctranspose` | Cheap check that `A'` is correctly implemented
`check_hermitian`  | Cheap check that `A = A'`
`check_positive_definite` | Cheap check that an operator is positive (semi-)definite


## Other Operations on Operators

Operators can be transposed (`A.'`), conjugated (`conj(A)`) and conjugate-transposed (`A'`).

## Other Operators

* [LLDL](https://github.com/optimizers/lldl) features a limited-memory
  LDL<sup>T</sup> factorization operator that may be used as preconditioner
  in iterative methods
* [MUMPS.jl](https://github.com/dpo/MUMPS.jl) features a full
  distributed-memory factorization operator that may be used to represent the
  preconditioner in, e.g., constraint-preconditioned Krylov methods.

## Testing

````JULIA
julia> Pkg.test("LinearOperators")
````

[![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)](http://www.gnu.org/licenses/gpl.html "GPLv3")
