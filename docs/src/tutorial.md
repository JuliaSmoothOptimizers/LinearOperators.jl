# Tutorial

This section of the documentation describes a few uses of LinearOperators.

```@contents
Pages = ["tutorial.md"]
```

## Using matrices

Operators may be defined from matrices and combined using the usual operations, but the result is deferred until the operator is applied.

```@example ex1
using LinearOperators, SparseArrays
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
using LinearAlgebra
A = rand(5,5)
A = A' * A
op = opCholesky(A)  # Use, e.g., as a preconditioner
v = rand(5)
norm(A \ v - op * v) / norm(v)
```
In this example, the Cholesky factor is computed only once and can be used many times transparently.

## mul!

It is often useful to reuse the memory used by the operator. 
For that reason, we can use `mul!` on operators as if we were using matrices
using preallocated vectors:

```@example ex2
using LinearOperators, LinearAlgebra # hide
m, n = 50, 30
A = rand(m, n) + im * rand(m, n)
op = LinearOperator(A)
v = rand(n)
res = zeros(eltype(A), m)
res2 = copy(res)
mul!(res2, op, v) # compile 3-args mul!
al = @allocated mul!(res, op, v) # op * v, store result in res
println("Allocation of LinearOperator mul! product = $al")
v = rand(n)
α, β = 2.0, 3.0
mul!(res2, op, v, α, β) # compile 5-args mul!
al = @allocated mul!(res, op, v, α, β) # α * op * v + β * res, store result in res
println("Allocation of LinearOperator mul! product = $al")
```

## Using functions

Operators may be defined from functions. They have to be based on the 5-arguments `mul!` function.
In the example below, the transposed isn't defined, but it may be inferred from the conjugate transposed. 
Missing operations are represented as `nothing`.
You will have deal with cases where `β == 0` and `β != 0` separately because `*` will allocate a `res` vector that
may contain `NaN` values, and `0 * NaN == NaN`.

```@example ex1
using FFTW
function mulfft!(res, v, α, β::T) where T
  if β == zero(T)
    res .= α .* fft(v)
  else
    res .= α .* fft(v) .+ β .* res
  end
end
function mulifft!(res, w, α, β::T) where T
  if β == zero(T)
    res .= α .* ifft(w)
  else
    res .= α .* ifft(w) .+ β .* res
  end
end
dft = LinearOperator(ComplexF64, 10, 10, false, false,
                     mulfft!,
                     nothing,       # will be inferred
                     mulifft!)
x = rand(10)
y = dft * x
norm(dft' * y - x)  # DFT is an orthogonal operator
```
```@example ex1
transpose(dft) * y
```

Another example:

```@example ex1
function customfunc!(res, v, α, β::T) where T
  if β == zero(T)
    res[1] = (v[1] + v[2]) * α
    res[2] = v[2] * α
  else
    res[1] = (v[1] + v[2]) * α + res[1] * β
    res[2] = v[2] * α + res[2] * β
  end
end
function tcustomfunc!(res, w, α, β::T) where T
  if β == zero(T)
    res[1] = w[1] * α
    res[2] =  (w[1] + w[2]) * α
  else
    res[1] = w[1] * α + res[1] * β
    res[2] =  (w[1] + w[2]) * α + res[2] * β
  end
end
op = LinearOperator(Float64, 10, 10, false, false,
                    customfunc!,
                    nothing,
                    tcustomfunc!)
```
Make sure that the type passed to `LinearOperator` is correct, otherwise errors may occur.
```@example ex1
using LinearOperators, FFTW # hide
dft = LinearOperator(Float64, 10, 10, false, false,
                     mulfft!,
                     nothing,
                     mulifft!)
v = rand(10)
println("eltype(dft)         = $(eltype(dft))")
println("eltype(v)           = $(eltype(v))")
# dft * v     # ERROR: expected Vector{Float64}
# Matrix(dft) # ERROR: tried to create a Matrix of Float64
``` 

## Limited memory BFGS and SR1

Two other useful operators are the Limited-Memory BFGS in forward and inverse form.

```@example ex1
B = LBFGSOperator(20)
H = InverseLBFGSOperator(20)
r = 0.0
for i = 1:100
  global r
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
Matrix(R)
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
