# A [Julia](http://julialang.org) Linear Operator Package

Operators behave like matrices but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector.

## Example 1

````JULIA
julia> A1 = rand(5,7);
julia> A2 = rand(7,3);
julia> op1 = LinearOperator(A1);
julia> op2 = LinearOperator(A2);
julia> op = op1 * op2;  # Does not form A1 * A2
julia> x = rand(3);
julia> y = op * x;
````

## Example 2

````JULIA
julia> A = rand(5,5); A = A' * A;
julia> op = opCholesky(A);  # Use, e.g., as a preconditioner
julia> v = rand(5);
julia> norm(A \ v - op * v) / norm(v)
1.6522645623951567e-14
````

## Testing

````JULIA
julia> Pkg.test("linop")
````

[![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)](http://www.gnu.org/licenses/gpl.html "GPLv3")
