# A [Julia](http://julialang.org) Linear Operator Package

[![DOI](https://zenodo.org/badge/20136006.svg)](https://zenodo.org/badge/latestdoi/20136006)
[![Build Status](https://travis-ci.org/JuliaSmoothOptimizers/LinearOperators.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/LinearOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/kp1o6ejuu6kgskvp/branch/master?svg=true)](https://ci.appveyor.com/project/dpo/linearoperators-jl/branch/master)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/LinearOperators.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/LinearOperators.jl)
[![Coveralls](https://coveralls.io/repos/JuliaSmoothOptimizers/LinearOperators.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaSmoothOptimizers/LinearOperators.jl?branch=master)
[![codecov.io](https://codecov.io/github/JuliaSmoothOptimizers/LinearOperators.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaSmoothOptimizers/LinearOperators.jl?branch=master)
[![docs](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/latest)

Cite as

    Dominique Orban, & Abel Soares Siqueira. (2019, February 7). LinearOperators.jl.
    Zenodo. http://doi.org/10.5281/zenodo.2559295

Operators behave like matrices (with some exceptions - see below) but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector.

## Compatibility

Julia 1.0 and up.

## How to Install

````JULIA
pkg> add LinearOperators
pkg> test LinearOperators
````

## How to use

Check the
[tutorial](https://JuliaSmoothOptimizers.github.io/LinearOperators.jl/latest/tutorial).

## Operators Available

Operator               | Description
-----------------------|------------
`LinearOperator`       | Base class. Useful to define operators from functions
`PreallocatedLinearOperator` | Linear operator with preallocated storage for products
`TimedLinearOperator`  | Linear operator instrumented with timers from [`TimerOutputs`](https://github.com/KristofferC/TimerOutputs.jl)
`BlockDiagonalOperator`| Block-diagonal linear operator
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
`check_ctranspose` | Cheap check that `A'` is correctly implemented
`check_hermitian`  | Cheap check that `A = A'`
`check_positive_definite` | Cheap check that an operator is positive (semi-)definite
`diag`             | Extract the diagonal of an operator
`full`             | Convert an abstract operator to a dense array
`hermitian`        | Determine whether the operator is Hermitian
`push!`            | For L-BFGS or L-SR1 operators, push a new pair {s,y}
`reset!`           | For L-BFGS or L-SR1 operators, reset the data
`shape`            | Return the size of a linear operator
`show`             | Display basic information about an operator
`size`             | Return the size of a linear operator
`symmetric`        | Determine whether the operator is symmetric


## Other Operations on Operators

Operators can be transposed (`transpose(A)`), conjugated (`conj(A)`) and conjugate-transposed (`A'`).
Operators can be sliced (`A[:,3]`, `A[2:4,1:5]`, `A[1,1]`), but unlike matrices, slices always return
operators (see differences below).

## Differences

Unlike matrices, an operator never reduces to a vector or a number.

````JULIA
A = rand(5,5)
opA = LinearOperator(A)
A[:,1] * 3 # Vector
opA[:,1] * 3 # LinearOperator
A[:,1] * [3] # ERROR
opA[:,1] * [3] # Vector
````

This is also true for `A[i,J]`, which returns vectors on 0.5, and for the scalar
`A[i,j]`.
Similarly, `opA[1,1]` is an operator of size (1,1):"
````JULIA
opA[1,1] # LinearOperator
A[1,1] # Number
````

In the same spirit, the operator `full` always returns a matrix.
````JULIA
full(opA[:,1]) # nx1 matrix
````


## Other Operators

* [LimitedLDLFactorizations](https://github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl) features a limited-memory
  LDL<sup>T</sup> factorization operator that may be used as preconditioner
  in iterative methods
* [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) features a full
  distributed-memory factorization operator that may be used to represent the
  preconditioner in, e.g., constraint-preconditioned Krylov methods.
