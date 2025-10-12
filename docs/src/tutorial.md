# Tutorial

This page follows the Documenter.jl tutorial layout. It includes runnable examples for common usage patterns in LinearOperators.jl.

```@contents
Pages = ["tutorial.md"]
```

## Getting started

Calling into LinearOperators is simple:

```@docs
using LinearOperators
```

## Simple examples

Construct an operator from a matrix and use it like a matrix:

```@example ex_matrix
using LinearOperators

A = [1.0 2.0; 3.0 4.0]
op = LinearOperator(A)
y = op * [1.0, 1.0]
M = Matrix(op)
println("y = ", y)
println("Matrix(op) = \n", M)
```

Create function-based operators (showing the 5-arg mul! signature):

```@example ex_fun
n = 4
function mymul!(res, v, α, β)
    if β == 0
        res .= α .* v .* (1:n)
    else
        res .= α .* v .* (1:n) .+ β .* res
    end
end
opfun = LinearOperator(Float64, n, n, false, false, mymul!)
println(opfun * ones(n))
```

Diagonal operators (use `opDiagonal`):

```@example ex_diag
d = [2.0, 3.0, 4.0]
D = opDiagonal(d)
println(D * ones(3))
```

Composing operators with vertical concatenation:

```@example ex_cat
A = rand(3,3); B = rand(3,3)
opA = LinearOperator(A); opB = LinearOperator(B)
opcat = [opA; opB]
println(size(opcat))
```

## Return values and tips

- The `LinearOperator` type implements `*` and can be converted to a `Matrix` when necessary.
- Prefer function-based operators when you want to avoid materializing large matrices.
- See the full introduction: [Introduction to LinearOperators.jl](https://jso.dev/tutorials/introduction-to-linear-operators/)
