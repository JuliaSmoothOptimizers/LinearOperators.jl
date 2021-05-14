# A Julia Linear Operator Package

Operators behave like matrices (with [exceptions](#Differences-1)) but are defined
by their effect when applied to a vector.
They can be transposed, conjugated, or combined with other operators cheaply.
The costly operation is deferred until multiplied with a vector.

## Compatibility

Julia 0.6 and up.

## How to Install

```julia
Pkg.add("LinearOperators")
```

## Operators Available

Operator                     | Description
-----------------------------|------------
`LinearOperator`             | Base class. Useful to define operators from functions
`TimedLinearOperator`        | Linear operator instrumented with timers from [`TimerOutputs`](https://github.com/KristofferC/TimerOutputs.jl)
`BlockDiagonalOperator`      | Block-diagonal linear operator
`opEye`                      | Identity operator
`opOnes`                     | All ones operator
`opZeros`                    | All zeros operator
`opDiagonal`                 | Square (equivalent to `diagm()`) or rectangular diagonal operator
`opInverse`                  | Equivalent to `\`
`opCholesky`                 | More efficient than `opInverse` for symmetric positive definite matrices
`opLDL`                      | Similar to `opCholesky`, for general sparse symmetric matrices
`opHouseholder`              | Apply a Householder transformation `I-2hh'`
`opHermitian`                | Represent a symmetric/hermitian operator based on the diagonal and strict lower triangle
`opRestriction`              | Represent a selection of "rows" when composed on the left with an existing operator
`opExtension`                | Represent a selection of "columns" when composed on the right with an existing operator
`LBFGSOperator`              | Limited-memory BFGS approximation in operator form (damped or not)
`InverseLBFGSOperator`       | Inverse of a limited-memory BFGS approximation in operator form (damped or not)
`LSR1Operator`               | Limited-memory SR1 approximation in operator form
`kron`                       | Kronecker tensor product in linear operator form

## Utility Functions

Function           | Description
-------------------|------------
`check_ctranspose` | Cheap check that `A'` is correctly implemented
`check_hermitian`  | Cheap check that `A = A'`
`check_positive_definite` | Cheap check that an operator is positive (semi-)definite
`diag`             | Extract the diagonal of an operator
`Matrix`           | Convert an abstract operator to a dense array
`hermitian`        | Determine whether the operator is Hermitian
`push!`            | For L-BFGS or L-SR1 operators, push a new pair {s,y}
`reset!`           | For L-BFGS or L-SR1 operators, reset the data
`shape`            | Return the size of a linear operator
`show`             | Display basic information about an operator
`size`             | Return the size of a linear operator
`symmetric`        | Determine whether the operator is symmetric


## Other Operations on Operators

Operators can be transposed (`A.'`), conjugated (`conj(A)`) and conjugate-transposed (`A'`).
Operators can be sliced (`A[:,3]`, `A[2:4,1:5]`, `A[1,1]`), but unlike matrices, slices always return
operators (see [differences](@ref differences)).

## [Differences](@id differences)

Unlike matrices, an operator never reduces to a vector or a number.

```@example exdiff
using LinearOperators #hide
A = rand(5,5)
opA = LinearOperator(A)
A[:,1] * 3 # Vector
```
```@example exdiff
opA[:,1] * 3 # LinearOperator
```
```@example exdiff
# A[:,1] * [3] # ERROR
```
```@example exdiff
opA[:,1] * [3] # Vector
```
This is also true for `A[i,:]`, which returns vectors on Julia 0.6, and for the scalar
`A[i,j]`.
Similarly, `opA[1,1]` is an operator of size (1,1):"
```@example exdiff
(opA[1,1] * [3])[1] - A[1,1] * 3
```

In the same spirit, the operator `Matrix` always returns a matrix.
```@example exdiff
Matrix(opA[:,1])
```

## Other Operators

* [LLDL](https://github.com/optimizers/lldl) features a limited-memory
  LDLᵀ factorization operator that may be used as preconditioner
  in iterative methods
* [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) features a full
  distributed-memory factorization operator that may be used to represent the
  preconditioner in, e.g., constraint-preconditioned Krylov methods.

## Testing

```julia
julia> Pkg.test("LinearOperators")
```
