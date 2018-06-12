# Tutorial

This section of the documentation describes a few uses of LinearOperators.

```@contents
Pages = ["tutorial.md"]
```

## Using matrices

Operators may be defined from matrices and combined using the usual operations, but the result is deferred until the operator is applied.

```@example ex1
using LinearOperators
A1 = rand(5,7)
A2 = sprand(7,3,.3)
op1 = LinearOperator(A1)
op2 = LinearOperator(A2)
op = op1 * op2  # Does not form A1 * A2
x = rand(3)
y = op * x
```

## Inverse

Operators may be defined to represent (approximate) inverses.

```@example ex1
A = rand(5,5)
A = A' * A
op = opCholesky(A)  # Use, e.g., as a preconditioner
v = rand(5)
norm(A \ v - op * v) / norm(v)
```
In this example, the Cholesky factor is computed only once and can be used many times transparently.

## Using functions

Operators may be defined from functions. In the example below, the transposed
isn't defined, but it may be inferred from the conjugate transposed. Missing
operations are represented as `nothing`.

```@example ex1
dft = LinearOperator(10, 10, false, false,
                     v -> fft(v),
                     nothing,       # will be inferred
                     w -> ifft(w))
x = rand(10)
y = dft * x
norm(dft' * y - x)  # DFT is an orthogonal operator
```
```@example ex1
dft.' * y
```

By default a linear operator defined by functions and that is neither symmetric
nor hermitian will have element type `Complex128`.
This behavior may be overridden by specifying the type explicitly, e.g.,
```@example ex1
dft = LinearOperator{Float64}(10, 10, false, false,
                              v -> fft(v),
                              nothing,
                              w -> ifft(w))
```

### Preallocation

It is also possible to use inplace functions by preallocating the required elements. For
instance:
```@example ex1
A = rand(5, 3)
Ap = Vector{Float64}(5)
Atq = Vector{Float64}(3)
op = LinearOperator(5, 3, false, false,
                    p -> A * p,
                    q -> A' * q)
v = op * ones(3)
norm(Ap - v)
```

The same can be done with functions:
```julia ex1
op = LinearOperator(5, 3, false, false,
                    p -> Aprod!(Ap, p)
                    q -> Atprod!(Atq, q))
```
where `Aprod!` and `Atprod!` return `Ap` and `Atq`, respectively.

## Limited memory BFGS and SR1

Two other useful operators are the Limited-Memory BFGS in forward and inverse form.

```@example ex1
B = LBFGSOperator(20)
H = InverseLBFGSOperator(20)
r = 0.0
for i = 1:100
  s = rand(20)
  y = rand(20)
  push!(B, s, y)
  push!(H, s, y)
  r += norm(B * H * s - s)
end
r
```

There is also a LSR1 operator that behaves similarly to these two.

## Restriction, extension and slices

The restriction operator restricts a vector to a set of indices.
```@example ex1
v = collect(1:5)
R = opRestriction([2;5], 5)
R * v
```
Notice that it corresponds to a matrix with rows of the identity given by the
indices.
```@example ex1
full(R)
```

The extension operator is the transpose of the restriction. It extends a vector
with zeros.
```@example ex1
v = collect(1:2)
E = opExtension([2;5], 5)
E * v
```

With these operators, we define the slices of an operator `op`.
```@example ex1
A = rand(5,5)
opA = LinearOperator(A)
I = [1;3;5]
J = 2:4
A[I,J] * ones(3)
```

```@example ex1
opRestriction(I, 5) * opA * opExtension(J, 5) * ones(3)
```

A main [difference](@ref differences) with matrices, is that slices **do not** return vectors nor
numbers.
```@example ex1
opA[1,:] * ones(5)
```
```@example ex1
opA[:,1] * ones(1)
```
```@example ex1
opA[1,1] * ones(1)
```
