# A [Julia](http://julialang.org) Linear Operator Package

[![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/LinearOperators.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/LinearOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/kp1o6ejuu6kgskvp/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/linearoperators-jl/branch/master)
[![Coveralls](https://coveralls.io/repos/JuliaSmoothOptimizers/LinearOperators.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaSmoothOptimizers/LinearOperators.jl?branch=master)
[![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/LinearOperators.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaSmoothOptimizers/LinearOperators.jl?branch=master)

Operators behave like matrices but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector.

## Compatibility

Julia 0.4 and up.

## How to Install

````JULIA
Pkg.add("LinearOperators")
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
In this example, the Cholesky factor is computed only once and can be used many times transparently.

## Example 3

Operators may be defined from functions. In the example below, the transposed isn't defined, but it may be inferred from the conjugate transposed. Missing operations are represented as [nullable](http://julia.readthedocs.org/en/latest/manual/types/?highlight=nullable#nullable-types-representing-missing-values) functions. Nullable types were introduced in Julia 0.4.

````JULIA
julia> dft = LinearOperator(10, 10, false, false,
                            v -> fft(v),
                            Nullable{Function}(),  # will be inferred
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

By default a linear operator defined by functions and that is neither symmetric nor hermitian will have element type `Complex128`.
This behavior may be overridden by specifying the type explicitly, e.g.,
```JULIA
dft = LinearOperator{Float64}(10, 10, false, false,
                              v -> fft(v),
                              Nullable{Function}(),
                              w -> ifft(w));
```

## Operators Available

Operator               | Description
-----------------------|------------
`LinearOperator`       | Base class. Useful to define operators from functions
`opEye`                | Identity operator
`opOnes`               | All ones operator
`opZeros`              | All zeros operator
`opDiagonal`           | Square (equivalent to `diagm()`) or rectangular diagonal operator
`opInverse`            | Equivalent to `\`
`opCholesky`           | More efficient than `opInverse` for symmetric positive definite matrices
`opHouseholder`        | Apply a Householder transformation `I-2hh'`
`opHermitian`          | Represent a symmetric/hermitian operator based on the diagonal and strict lower triangle
`opRestriction`        | Represent a selection of "rows" when composed on the left with an existing operator
`opExtension`          | Represent a selection of "columns" when composed on the right with an existing operator
`LBFGSOperator`        | Limited-memory BFGS approximation in operator form (damped or not)
`InverseLBFGSOperator` | Inverse of a limited-memory BFGS approximation in operator form (damped or not)
`LSR1Operator`         | Limited-memory SR1 approximation in operator form

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
* [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) features a full
  distributed-memory factorization operator that may be used to represent the
  preconditioner in, e.g., constraint-preconditioned Krylov methods.

## Testing

````JULIA
julia> Pkg.test("LinearOperators")
````

[![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)](http://www.gnu.org/licenses/gpl.html "GPLv3")
