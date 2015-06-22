# Linear Operators for Julia
module LinearOperators

using Compat  # for Nullable types.

# Setup for documentation
using Docile

export AbstractLinearOperator,
       LinearOperator, opEye, opOnes, opZeros, opDiagonal,
       opInverse, opCholesky, opHouseholder, opHermitian,
       LBFGSOperator, InverseLBFGSOperator,
       check_ctranspose, check_hermitian, check_positive_definite,
       shape, hermitian, symmetric

KindOfMatrix = Union(Array, SparseMatrixCSC)

abstract AbstractLinearOperator;


Docile.@doc """Abstract object to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
""" ->
type LinearOperator <: AbstractLinearOperator
  nrow   :: Int
  ncol   :: Int
  dtype   :: DataType
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
end


import Base.size

Docile.@doc meta("Return the size of a linear operator as a tuple", returns=(Int,Int)) ->
size(op :: AbstractLinearOperator) = (op.nrow, op.ncol)

Docile.@doc meta("Return the size of a linear operator along dimension `d`", returns=(Int,)) ->
function size(op :: AbstractLinearOperator, d :: Int)
  if d == 1
    return op.nrow;
  end
  if d == 2
    return op.ncol;
  end
  error("Linear operators only have 2 dimensions for now");
end

Docile.@doc "An alias for size" ->
shape(op :: AbstractLinearOperator) = size(op)

Docile.@doc meta("Determine whether the operator is Hermitian", returns=(Bool,)) ->
hermitian(op :: AbstractLinearOperator) = op.hermitian

Docile.@doc meta("Determine whether the operator is symmetric", returns=(Bool,)) ->
symmetric(op :: AbstractLinearOperator) = op.symmetric


import Base.show

Docile.@doc "Display basic information about a linear operator" ->
function show(io :: IO, op :: AbstractLinearOperator)
  s  = "Linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  dtype: %s\n", op.dtype)
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  prod:   %s\n", string(op.prod))
  s *= @sprintf("  tprod:  %s\n", string(op.tprod))
  s *= @sprintf("  ctprod: %s", string(op.ctprod))
  s *= "\n"
  print(io, s)
end


# Constructors.
Docile.@doc """Construct a linear operator from a dense or sparse matrix.
Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.""" ->
LinearOperator(M :: KindOfMatrix; symmetric=false, hermitian=false) =
  LinearOperator(size(M,1), size(M,2), typeof(M[1,1]), symmetric, hermitian,
                 v -> M * v,
                 Nullable{Function}(u -> M.' * u),
                 Nullable{Function}(w -> M' * w))

Docile.@doc "Construct a linear operator from functions." ->
LinearOperator(nrow :: Int, ncol :: Int, dtype :: DataType,
               symmetric :: Bool, hermitian :: Bool,
               prod :: Function, tprod :: Function, ctprod :: Function) =
  LinearOperator(nrow, ncol, dtype, symmetric, hermitian,
                 prod, Nullable{Function}(tprod), Nullable{Function}(ctprod))

Docile.@doc "Construct a real symmetric linear operator from a function." ->
LinearOperator(nrow :: Int, dtype :: DataType, prod :: Function) =
  LinearOperator(nrow, nrow, dtype, true, true,
                 prod,
                 Nullable{Function}(prod),
                 Nullable{Function}(prod))

Docile.@doc "Construct a linear operator from a single function." ->
LinearOperator(nrow :: Int, ncol :: Int, dtype :: DataType,
               symmetric :: Bool, hermitian :: Bool,
               prod :: Function) =
  LinearOperator(nrow, ncol, dtype, symmetric, hermitian,
                 prod, Nullable{Function}(), Nullable{Function}())


# Apply an operator to a vector.
function (*)(op :: AbstractLinearOperator, v :: Vector)
  (m, n) = size(op)
  if size(v,1) != n
    error("Shape mismatch")
  end
  return op.prod(v)
end

import Base.full

Docile.@doc "Materialize an operator as a dense array using `op.ncol` products" ->
function full(op :: AbstractLinearOperator)
  (m, n) = size(op)
  A = zeros(op.dtype, m, n)  # Must be of same dtype as operator.
  ei = zeros(op.dtype, n)
  for i = 1 : n
    ei[i] = 1;
    A[:,i] = op * ei;
    ei[i] = 0;
  end
  return A
end


# Unary operations.
(+)(op :: AbstractLinearOperator) = op
(-)(op :: AbstractLinearOperator) = LinearOperator(op.nrow, op.ncol, op.dtype,
                                                   op.symmetric, op.hermitian,
                                                   v -> -op.prod(v),
                                                   Nullable{Function}(u -> -get(op.tprod)(u)),
                                                   Nullable{Function}(w -> -get(op.ctprod)(w)))

function transpose(op :: AbstractLinearOperator)
  if op.symmetric
    return op
  end
  if !isnull(op.tprod)
    return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                          get(op.tprod),
                          Nullable{Function}(op.prod),
                          Nullable{Function}(v -> conj(get(op.tprod)(v))))
  end
  if isnull(op.ctprod)
    if op.hermitian
      ctprod = op.prod;
    else
      error("Unable to infer transpose operator")
    end
  else
    ctprod = get(op.ctprod);
  end

  return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(ctprod(conj(v))),                # A.'v = conj(A' conj(v))
                        Nullable{Function}(op.prod),               # (A.').' = A
                        Nullable{Function}(w -> conj(op.prod(w)))) # (A.')' = conj(A)
end

function ctranspose(op :: LinearOperator)
  if op.hermitian
    return op
  end
  if !isnull(op.ctprod)
    return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                          get(op.ctprod),
                          Nullable{Function}(u -> conj(op.prod(u))),
                          Nullable{Function}(op.prod))
  end
  if isnull(op.tprod)
    if op.symmetric
      tprod = op.prod;
    else
      error("Unable to infer conjugate transpose operator")
    end
  else
    tprod = get(op.tprod);
  end

  return LinearOperator(op.ncol, op.nrow, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(tprod(v)),
                        Nullable{Function}(u -> conj(op.prod(u))),
                        Nullable{Function}(op.prod))
end

import Base.conj
function conj(op :: AbstractLinearOperator)
  return LinearOperator(op.nrow, op.ncol, op.dtype, op.symmetric, op.hermitian,
                        v -> conj(op.prod(conj(v))),
                        op.ctprod,
                        op.tprod)
end

# Binary operations.

## Operator times operator.
function (*)(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    error("Shape mismatch")
  end
  result_type = promote_type(op1.dtype, op2.dtype)
  return LinearOperator(m1, n2, result_type, false, false,
                        v -> op1 * (op2 * v),
                        u -> op2.' * (op1.' * u),
                        w -> op2' * (op1' * w))
end

## Matrix times operator.
(*)(M :: KindOfMatrix, op :: AbstractLinearOperator) = LinearOperator(M) * op
(*)(op :: AbstractLinearOperator, M :: KindOfMatrix) = op * LinearOperator(M)

## Scalar times operator.
(*)(op :: AbstractLinearOperator, x :: Number) = LinearOperator(op.nrow, op.ncol,
                                                                promote_type(op.dtype, typeof(x)),
                                                                op.symmetric,
                                                                op.hermitian && isreal(x),
                                                                v -> (op * v) * x,
                                                                u -> x * (op.' * u),
                                                                w -> x' * (op' * w))
(*)(x :: Number, op :: AbstractLinearOperator) = LinearOperator(op.nrow, op.ncol,
                                                                promote_type(op.dtype, typeof(x)),
                                                                op.symmetric,
                                                                op.hermitian && isreal(x),
                                                                v -> x * (op * v),
                                                                u -> (op.' * u) * x,
                                                                w -> (op' * w) * x')
(.*)(op :: AbstractLinearOperator, x :: Number) = op * x
(.*)(x :: Number, op :: AbstractLinearOperator) = x * op

# Operator + operator.
function (+)(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    error("Shape mismatch")
  end
  return LinearOperator(m1, n1, promote_type(op1.dtype, op2.dtype),
                        op1.symmetric && op2.symmetric,
                        op1.hermitian && op2.hermitian,
                        v -> (op1   * v) + (op2   * v),
                        u -> (op1.' * u) + (op2.' * u),
                        w -> (op1'  * w) + (op2'  * w))
end

# Operator + matrix.
(+)(M :: KindOfMatrix, op :: AbstractLinearOperator) = LinearOperator(M) + op
(+)(op :: AbstractLinearOperator, M :: KindOfMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
(.+)(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
(.+)(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
(-)(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
(-)(M :: KindOfMatrix, op :: AbstractLinearOperator) = LinearOperator(M) - op
(-)(op :: AbstractLinearOperator, M :: KindOfMatrix) = op - LinearOperator(M)

# Operator - scalar.
(.-)(op :: AbstractLinearOperator, x :: Number) = op .+ (-x)
(.-)(x :: Number, op :: AbstractLinearOperator) = x .+ (-op)


# Utility functions.

Docile.@doc "Cheap check that the operator and its conjugate transposed are related." ->
function check_ctranspose(op :: AbstractLinearOperator)
  (m, n) = size(op);
  x = rand(n);
  y = rand(m);
  yAx = dot(y, op * x);
  xAty = dot(x, op' * y);
  ε = eps(Float64);
  return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1/3);
end

check_ctranspose(M :: KindOfMatrix) = check_ctranspose(LinearOperator(M))

Docile.@doc "Cheap check that the operator is Hermitian." ->
function check_hermitian(op :: AbstractLinearOperator)
  m, n = size(op);
  v = rand(n);
  w = op * v;
  s = dot(w, w);  # = (Av)'(Av) = v' A' A v.
  y = op * w;
  t = dot(v, y);  # = v' A A v.
  ε = eps(Float64);
  return abs(s - t) < (abs(s) + ε) * ε^(1/3);
end

check_hermitian(M :: KindOfMatrix) = check_hermitian(LinearOperator(M))

Docile.@doc "Cheap check that the operator is positive (semi-)definite." ->
function check_positive_definite(op :: AbstractLinearOperator; semi=false)
  m, n = size(op);
  v = rand(n);
  w = op * v;
  vw = dot(v, w);
  ε = eps(Float64);
  if imag(vw) > sqrt(ε) * abs(vw)
    return false
  end
  vw = real(vw);
  return semi ? (vw ≥ 0) : (vw > 0)
end

check_positive_definite(M :: KindOfMatrix) = check_positive_definite(LinearOperator(M))

# Special linear operators.

Docile.@doc "Identity operator of order `n` and of data type `dtype`." ->
opEye(n :: Int; dtype=Float64) = LinearOperator(n, n, dtype, true, true,
                                                v -> v[:], u -> u[:], w -> w[:])

Docile.@doc "Operator of all ones of size `nrow`-by-`ncol` and of data type `dtype`." ->
opOnes(nrow, ncol; dtype=Float64) = LinearOperator(nrow, ncol, dtype,
                                                   nrow == ncol, nrow == ncol,
                                                   v -> sum(v) * ones(nrow),
                                                   u -> sum(u) * ones(ncol),
                                                   w -> sum(w) * ones(ncol))

Docile.@doc "Zero operator of size `nrow`-by-`ncol` and of data type `dtype`." ->
opZeros(nrow, ncol; dtype=Float64) = LinearOperator(nrow, ncol, dtype,
                                                    nrow == ncol, nrow == ncol,
                                                    v -> zeros(nrow),
                                                    u -> zeros(ncol),
                                                    w -> zeros(ncol))

Docile.@doc "Diagonal operator with the vector `d` on its main diagonal." ->
opDiagonal(d :: Vector) = LinearOperator(length(d), length(d), typeof(d[1]),
                                         true, !(typeof(d[1]) <: Complex),
                                         v -> v .* d,
                                         u -> u .* d,
                                         w -> w .* conj(d))

Docile.@doc """Rectangular diagonal operator of size `nrow`-by-`ncol`
with the vector `d` on its main diagonal.""" ->
function opDiagonal(nrow :: Int, ncol :: Int, d :: Vector)
  if nrow == ncol
    return opDiagonal(d)
  end
  if nrow > ncol
    D = LinearOperator(nrow, ncol, typeof(d[1]), false, false,
                       v -> [v .* d ; zeros(nrow-ncol)],
                       u -> u[1:ncol] .* d,
                       w -> w[1:ncol] .* conj(d));
  else
    D = LinearOperator(nrow, ncol, typeof(d[1]), false, false,
                       v -> v[1:nrow] .* d,
                       u -> [u .* d ; zeros(ncol-nrow)],
                       w -> [w .* conj(d) ; zeros(ncol-nrow)]);
  end
  return D
end


import Base.hcat
function hcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.nrow != B.nrow && error("hcat: inconsistent row sizes")

  nrow  = A.nrow
  ncol  = A.ncol + B.ncol
  dtype = promote_type(A.dtype, B.dtype)

  prod(v)   =  A * v[1:A.ncol] + B * v[A.ncol+1:end]
  tprod(v)  =  [A.' * v; B.' * v;]
  ctprod(v) =  [A' * v; B' * v;]

  return LinearOperator(nrow, ncol, A.dtype, false, false, prod, tprod, ctprod)
end

function hcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]];
  end
  return op
end

import Base.vcat

function vcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.ncol != B.ncol && error("vcat: inconsistent column sizes")

  nrow  = A.nrow + B.nrow
  ncol  = A.ncol
  dtype = promote_type(A.dtype, B.dtype)

  prod(v)   =  [A * v; B * v;]
  tprod(v)  =  A.' * v +  B.' * v
  ctprod(v) =  A' * v[1:A.nrow] + B' * v[A.nrow+1:end]

  return LinearOperator(nrow, ncol, dtype, false, false, prod, tprod, ctprod)
end

function vcat(ops :: AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]];
  end
  return op
end


Docile.@doc """Inverse of a matrix as a linear operator using `\`.
Useful for triangular matrices. Note that each application of this
operator applies `\`.""" ->
opInverse(M :: KindOfMatrix; symmetric=false, hermitian=false) =
  LinearOperator(size(M,2), size(M,1), typeof(M[1,1]), symmetric, hermitian,
                 v -> M \ v, u -> M.' \ u, w -> M' \ w);

Docile.@doc """Inverse of a positive definite matrix as a linear operator
using its Cholesky factorization. The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks. If the input is sparse and not positive definite, but possesses a
LDL' factorization, the latter is computed.""" ->
function opCholesky(M :: KindOfMatrix; check=false)
  (m, n) = size(M)
  if m != n
    error("Shape mismatch")
  end
  if check
    check_hermitian(M) || error("Matrix is not Hermitian")
    check_positive_definite(M) || error("Matrix is not positive definite")
  end
  if issparse(M)
    LDL = cholfact(M);
    return LinearOperator(m, m, typeof(M[1,1]),
                          !(typeof(M[1,1]) <: Complex), true,
                          v -> LDL \ v,
                          u -> conj(LDL \ conj(u)),  # M.' = conj(M)
                          w -> LDL \ w)
  else
    L = chol(M, :L);
    return LinearOperator(m, m, typeof(M[1,1]),
                          !(typeof(M[1,1]) <: Complex), true,
                          v -> L' \ (L \ v),
                          u -> L.' \ (conj(L \ conj(u))),
                          w -> L' \ (L \ w))
  end
  # Todo: use iterative refinement.
end

Docile.@doc """Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h h') x`.""" ->
opHouseholder(h :: Vector) = LinearOperator(length(h), length(h), typeof(h[1]),
                                            !(typeof(h[1]) <: Complex), true,
                                            v -> (v - 2 * dot(h, v) * h),
                                            Nullable{Function}(),  # Will be inferred.
                                            w -> (w - 2 * dot(h, w) * h))



Docile.@doc "A symmetric/hermitian operator based on the diagonal and lower triangle." ->
function opHermitian(d :: Vector, T :: KindOfMatrix)
  L = tril(T, -1);
  return LinearOperator(length(d), length(d), typeof(d[1]),
                        !(typeof(d[1]) <: Complex), true,
                        v -> (d .* v + L * v + (v' * L)')[:],
                        Nullable{Function}(),
                        Nullable{Function}());
end


Docile.@doc "A symmetric/hermitian operator based on a matrix." ->
function opHermitian(T :: KindOfMatrix)
  d = diag(T);
  return opHermitian(d, T);
end


Docile.@doc "A data type to hold information relative to LBFGS operators." ->
type LBFGSData
  mem :: Int;
  scaling :: Bool;
  s   :: Array;
  y   :: Array;
  ys  :: Vector;
  α   :: Vector;
  a   :: Array;
  b   :: Array;
  insert :: Int;

  function LBFGSData(n :: Int, mem :: Int;
                     dtype :: DataType=Float64, scaling :: Bool=false, inverse :: Bool=true)
    return new(max(mem, 1),
               scaling,
               zeros(dtype, n, mem),
               zeros(dtype, n, mem),
               zeros(dtype, mem),
               inverse ? zeros(dtype, mem) : dtype[],
               inverse ? dtype[] : zeros(dtype, n, mem),
               inverse ? dtype[] : zeros(dtype, n, mem),
               1)
  end
end


Docile.@doc "A type for limited-memory BFGS approximations." ->
type LBFGSOperator <: AbstractLinearOperator
  nrow   :: Int
  ncol   :: Int
  dtype   :: DataType
  symmetric :: Bool
  hermitian :: Bool
  prod   :: Function           # apply the operator to a vector
  tprod  :: Nullable{Function} # apply the transpose operator to a vector
  ctprod :: Nullable{Function} # apply the transpose conjugate operator to a vector
  inverse :: Bool
  data :: LBFGSData
end

Docile.@doc "Construct a limited-memory BFGS approximation in inverse form." ->
function InverseLBFGSOperator(n, mem :: Int=5; dtype :: DataType=Float64, scaling :: Bool=false)
  lbfgs_data = LBFGSData(n, mem, dtype=dtype, scaling=scaling);
  insert = 1;

  function lbfgs_multiply(data :: LBFGSData, x :: Array)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.4, p. 178.

    if dtype == typeof(x[1])
      q = copy(x);
    else
      result_type = promote_type(dtype, typeof(x[1]))
      q = convert(Array{result_type}, x);
    end

    for i = 1 : data.mem
      k = mod(data.insert - i - 1, data.mem) + 1;
      if data.ys[k] != 0
        data.α[k] = dot(data.s[:,k], q) / data.ys[k];
        q -= data.α[k] * data.y[:,k];
      end
    end

    r = q;
    if data.scaling
      last = mod(data.insert -1, data.mem) + 1;
      if data.ys[last] != 0
        γ = data.ys[last] / dot(data.y[:,last], data.y[:,last]);
        r *= γ
      end
    end

    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1;
      if data.ys[k] != 0
        β = dot(data.y[:,k], r) / data.ys[k];
        r += (data.α[k] - β) * data.s[:,k];
      end
    end

    return r
  end

  return LBFGSOperator(n, n, dtype, true, true,
                       x -> lbfgs_multiply(lbfgs_data, x),
                       Nullable{Function}(),
                       Nullable{Function}(),
                       true,
                       lbfgs_data)
end

Docile.@doc "Construct a limited-memory BFGS approximation in forward form." ->
function LBFGSOperator(n, mem :: Int=5; dtype :: DataType=Float64, scaling :: Bool=false)
  lbfgs_data = LBFGSData(n, mem, dtype=dtype, scaling=scaling, inverse=false);
  insert = 1;

  function lbfgs_multiply(data :: LBFGSData, x :: Array)
    # Multiply operator with a vector.
    # See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.6, p. 184.

    if dtype == typeof(x[1])
      q = copy(x);
    else
      result_type = promote_type(dtype, typeof(x[1]))
      q = convert(Array{result_type}, x);
    end

    # B = B₀ + Σᵢ (bᵢbᵢ' - aᵢaᵢ').
    for i = 1 : data.mem
      k = mod(data.insert + i - 2, data.mem) + 1;
      if data.ys[k] != 0
        q += dot(data.b[:, k], x) * data.b[:, k] - dot(data.a[:, k], x) * data.a[:, k];
      end
    end
    return q
  end

  return LBFGSOperator(n, n, dtype, true, true,
                       x -> lbfgs_multiply(lbfgs_data, x),
                       Nullable{Function}(),
                       Nullable{Function}(),
                       false,
                       lbfgs_data)
end

import Base.push!

Docile.@doc "Push a new {s,y} pair into a L-BFGS operator." ->
function push!(op :: LBFGSOperator, s :: Vector, y :: Vector)

  ys = dot(y, s);
  if ys <= 1.0e-20
    # warn(@sprintf("Rejecting L-BFGS {s,y} pair: y's = %8.1e", ys))
    return
  end

  data = op.data;
  insert = data.insert;

  data.s[:, insert] = s;
  data.y[:, insert] = y;
  data.ys[insert] = ys;

  # Update arrays a and b used in forward products.
  if !op.inverse
    data.b[:, insert] = y / sqrt(ys);

    for i = 1 : data.mem
      k = mod(insert + i - 1, data.mem) + 1;
      if data.ys[k] != 0
        data.a[:, k] = data.s[:, k];   # B₀ = I.

        for j = 1 : i - 1
          l = mod(insert + j - 1, data.mem) + 1;
          if data.ys[l] != 0
            data.a[:, k] += dot(data.b[:, l], data.s[:, k]) * data.b[:, l];
            data.a[:, k] -= dot(data.a[:, l], data.s[:, k]) * data.a[:, l];
          end
        end
        data.a[:, k] /= sqrt(dot(data.s[:, k], data.a[:, k]));
      end
    end
  end

  op.data.insert = mod(insert, data.mem) + 1;
  return
end

end  # module

