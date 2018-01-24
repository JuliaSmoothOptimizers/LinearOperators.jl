using Compat.LinearAlgebra, Compat.SparseArrays

function test_linop()
  (nrow, ncol) = (10, 6);
  ϵ = eps(Float64);
  rtol = sqrt(ϵ);
  A1 = rand(nrow, ncol) + rand(nrow, ncol) * im;

  op = LinearOperator(A1);
  show(op);

  # Test data type.
  @test eltype(op) == eltype(A1)
  @test !isreal(op)

  # Test size().
  @test(size(op) == (nrow, ncol));
  @test(shape(op) == (nrow, ncol));
  @test(size(op, 1) == nrow);
  @test(size(op, 2) == ncol);
  @test_throws LinearOperatorException size(op, 3)
  @test_throws LinearOperatorException op * rand(ncol + 1)

  # Test boolean operators.
  @test(symmetric(op) == false);
  @test(hermitian(op) == false);

  # Test full().
  @test(vecnorm(A1 - full(op)) <= ϵ * vecnorm(A1));

  # Test unary +.
  @test(vecnorm(A1 - full(+op)) <= ϵ * vecnorm(A1));

  # Test LinearOperator(Matrix).
  A2 = sprand(nrow, ncol, 0.5) + sprand(nrow, ncol, 0.5) * im;

  for A in (A1, A2)
    op = LinearOperator(A);
    @test(op.nrow == nrow);
    @test(op.ncol == ncol);

    @test(vecnorm(A   - full(op))   <= rtol * vecnorm(A));
    @test(vecnorm(transpose(A) - full(transpose(op))) <= rtol * vecnorm(A));
    @test(vecnorm(A'  - full(op'))  <= rtol * vecnorm(A));

    v = rand(ncol) + rand(ncol) * im;
    @test(norm(A * v - op * v) <= rtol * norm(v));

    u = rand(nrow) + rand(nrow) * im;
    @test(norm(transpose(A) * u - transpose(op) * u) <= rtol * norm(u));
    @test(norm(A'  * u - op'  * u) <= rtol * norm(u));
  end

  # Test basic arithmetic operations.
  B1 = rand(nrow, ncol) + rand(nrow, ncol) * im;

  for q in (+, -)
    C = q(A1, B1);
    opC = q(LinearOperator(A1), LinearOperator(B1));
    v = rand(ncol) + rand(ncol) * im;
    @test(norm(opC * v - C * v) <= rtol * norm(v));
    u = rand(nrow) + rand(nrow) * im;
    @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u));
    @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));

    opC = q(A1, LinearOperator(B1));
    v = rand(ncol) + rand(ncol) * im;
    @test(norm(opC * v - C * v) <= rtol * norm(v));
    u = rand(nrow) + rand(nrow) * im;
    @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u));
    @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));

    opC = q(LinearOperator(A1), B1);
    v = rand(ncol) + rand(ncol) * im;
    @test(norm(opC * v - C * v) <= rtol * norm(v));
    u = rand(nrow) + rand(nrow) * im;
    @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u));
    @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));
  end

  # Operator +/- scalar.
  opC = LinearOperator(A1) .+ 2.12345;
  @test(vecnorm(A1 .+ 2.12345 - full(opC)) <= rtol * vecnorm(A1 .+ 2.12345));

  opC = 2.12345 .+ LinearOperator(A1);
  @test(vecnorm(A1 .+ 2.12345 - full(opC)) <= rtol * vecnorm(A1 .+ 2.12345));

  opC = LinearOperator(A1) .- 2.12345;
  @test(vecnorm((A1 .- 2.12345) - full(opC)) <= rtol * vecnorm(A1 .- 2.12345));

  opC = 2.12345 .- LinearOperator(A1);
  @test(vecnorm((2.12345 .- A1) - full(opC)) <= rtol * vecnorm(2.12345 .- A1));

  B2 = rand(ncol, ncol+1) + rand(ncol, ncol+1) * im;
  C = A1 * B2;
  opC = LinearOperator(A1) * LinearOperator(B2);
  v = rand(ncol+1) + rand(ncol+1) * im;
  @test(norm(opC * v - C * v) <= rtol * norm(v));
  u = rand(nrow) + rand(nrow) * im;
  @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u));
  @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));

  @test_throws LinearOperatorException LinearOperator(A1) + LinearOperator(B2);
  @test_throws LinearOperatorException LinearOperator(B2) * LinearOperator(A1);

  # Matrix * operator.
  A1B2 = A1 * B2;
  opC = A1 * LinearOperator(B2);
  @test(vecnorm(A1B2 - full(opC)) <= rtol * vecnorm(A1B2));

  # Operator * matrix.
  opC = LinearOperator(A1) * B2;
  @test(vecnorm(A1B2 - full(opC)) <= rtol * vecnorm(A1B2));

  # Scalar * operator.
  AA1 = 2.12345 * A1;
  opC = 2.12345 * LinearOperator(A1);
  @test(vecnorm(AA1 - full(opC)) <= rtol * vecnorm(AA1));

  opC = 2.12345 .* LinearOperator(A1);
  @test(vecnorm(AA1 - full(opC)) <= rtol * vecnorm(AA1));

  # Operator * scalar.
  opC = LinearOperator(A1) * 2.12345;
  @test(vecnorm(AA1 - full(opC)) <= rtol * vecnorm(AA1));

  opC = LinearOperator(A1) .* 2.12345;
  @test(vecnorm(AA1 - full(opC)) <= rtol * vecnorm(AA1));

  # Test Cholesky operator.
  AA = A1' * A1;
  AAinv = opCholesky(AA, check=true);
  v = rand(ncol) + im * rand(ncol);
  @test(norm(AAinv * v - AA \ v) <= rtol * norm(v));

  # Test Hermitian operator.
  A1 = rand(nrow, nrow) + rand(nrow, nrow) * im;
  d = diag(A1); A1 = tril(A1, -1);
  A2 = A1 + A1' + diagm(0 => d);
  op = opHermitian(d, A1);
  v  = rand(nrow) + rand(nrow) * im;
  Av = A2 * v;
  @test(norm(op * v - Av) <= rtol * norm(Av));

  # Test opEye.
  I = opEye(nrow);
  v = rand(nrow) + rand(nrow) * im;
  @test(abs(norm(I * v - v)) <= ϵ * norm(v));
  @test(abs(norm(transpose(I) * v - v)) <= ϵ * norm(v));
  @test(abs(norm(I' * v - v)) <= ϵ * norm(v));
  @test(vecnorm(full(I) - eye(nrow)) <= ϵ * vecnorm(eye(nrow)));

  I = opEye(nrow, ncol)
  v = rand(ncol) + rand(ncol) * im
  v0 = [v ; zeros(nrow - ncol)]
  vu = [v ; rand(nrow - ncol)]
  @test(abs(norm(I * v - v0)) <= ϵ * norm(v))
  @test(abs(norm(transpose(I) * vu - v)) <= ϵ * norm(v))
  @test(abs(norm(I' * vu - v)) <= ϵ * norm(v))
  @test(vecnorm(full(I) - eye(nrow, ncol)) <= ϵ * vecnorm(eye(nrow, ncol)))

  I = opEye(ncol, nrow)
  @test(abs(norm(I * vu - v)) <= ϵ * norm(v))
  @test(abs(norm(transpose(I) * v - v0)) <= ϵ * norm(v))
  @test(abs(norm(I' * v - v0)) <= ϵ * norm(v))
  @test(vecnorm(full(I) - eye(ncol, nrow)) <= ϵ * vecnorm(eye(ncol, nrow)))

  # Test opOnes.
  E = opOnes(nrow, ncol);
  v  = rand(nrow) + rand(nrow) * im;
  u = rand(ncol) + rand(ncol) * im;
  @test(norm(E * u - sum(u) * ones(nrow)) <= rtol * norm(u));
  @test(norm(transpose(E) * v - sum(v) * ones(ncol)) <= rtol * norm(v));
  @test(norm(E' * v - sum(v) * ones(ncol)) <= rtol * norm(v));

  # Test opZeros.
  O = opZeros(nrow, ncol);
  @test(norm(O * u) <= ϵ);
  @test(norm(transpose(O) * v) <= ϵ);
  @test(norm(O' * v) <= ϵ);

  # Test opDiagonal.
  D = opDiagonal(v);
  u = rand(nrow) + rand(nrow) * im;
  @test(norm(D * u - v .* u) <= ϵ * norm(u));
  @test(norm(transpose(D) * u - v .* u) <= ϵ * norm(u));
  @test(norm(D' * u - conj(v) .* u) <= ϵ * norm(u));

  # Test rectangular opDiagonal.
  nmin = min(nrow, ncol); nmax = max(nrow, ncol);
  A = zeros(ComplexF64, nmax, nmin);
  v = rand(nmin) + rand(nmin) * im;
  for i = 1 : nmin
    A[i,i] = v[i];
  end
  D = opDiagonal(nmax, nmin, v);
  u = rand(nmin) + rand(nmin) * im;
  @test(norm(A * u - D * u) <= ϵ * norm(u));
  w = rand(nmax) + rand(nmax) * im;
  @test(norm(transpose(A) * w - transpose(D) * w) <= ϵ * norm(w));
  @test(norm(A' * w - D' * w) <= ϵ * norm(w));

  A = zeros(ComplexF64, nmin, nmax);
  for i = 1 : nmin
    A[i,i] = v[i];
  end
  D = opDiagonal(nmin, nmax, v);
  @test(norm(A * w - D * w) <= ϵ * norm(w));
  @test(norm(transpose(A) * u - transpose(D) * u) <= ϵ * norm(u));
  @test(norm(A' * u - D' * u) <= ϵ * norm(u));

  # Test opInverse.
  (U, _) = qr(rand(nrow, nrow) + rand(nrow, nrow) * im);
  (V, _) = qr(rand(nrow, nrow) + rand(nrow, nrow) * im);
  Σ = diagm(0 => rand(nrow) + 0.1);
  A = U * Σ * V';
  Ainv = opInverse(A);
  v = rand(nrow) + rand(nrow) * im;
  @test(norm(A \ v - Ainv * v) <= rtol * norm(v));
  @test(norm(transpose(A) \ v - transpose(Ainv) * v) <= rtol * norm(v));
  @test(norm(A' \ v - Ainv' * v) <= rtol * norm(v));

  # Test opCholesky.
  B = A' * A;
  Binv = opCholesky(B, check=true);
  @test(norm(B \ v - Binv * v) <= rtol * norm(v));
  @test(norm(transpose(B) \ v - transpose(Binv) * v) <= rtol * norm(v));
  @test(norm(B' \ v - Binv' * v) <= rtol * norm(v));

  @test_throws LinearOperatorException opCholesky(rand(3,5));
  @test_throws LinearOperatorException opCholesky(rand(5,5), check=true);

  # Test opHouseholder.
  H = opHouseholder(v);
  u = rand(nrow) + rand(nrow) * im;
  @test(norm(H * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u));
  @test(norm(transpose(H) * u - (u - 2 * dot(conj(v), u) * conj(v))) <= rtol * norm(u));
  @test(norm(H' * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u));

  # Test opHermitian.
  C = A + A';
  H = opHermitian(C);
  @test(norm(H * v - C * v) <= rtol * norm(v));
  @test(norm(transpose(H) * v - transpose(C) * v) <= rtol * norm(v));
  @test(norm(H' * v - C * v) <= rtol * norm(v));

  @test(! check_hermitian(LinearOperator(A - A')));
  @test(! check_positive_definite(LinearOperator(-A'*A)));

  # Test inference.
  op = LinearOperator(5, 3, false, false,
                      p -> ones(5) + im * ones(5));
  @test eltype(op) == ComplexF64
  @test_throws LinearOperatorException transpose(op)  # cannot be inferred
  @test_throws LinearOperatorException op'            # cannot be inferred

  op2 = conj(op);
  @test(vecnorm(full(op2) - conj(full(op))) <= ϵ * vecnorm(full(op)));

  A = rand(5,3) + im * rand(5,3);
  op = LinearOperator(A);
  @test(check_ctranspose(A));
  @test(check_ctranspose(op));
  @test_throws LinearOperatorException opCholesky(A)  # Shape mismatch

  A = rand(5,5) + im * rand(5,5);
  @test_throws LinearOperatorException opCholesky(A, check=true)  # Not Hermitian / positive definite
  @test_throws LinearOperatorException opCholesky(-A'*A, check=true)  # Not positive definite

  # Test Cholesky operator on SQD matrix.
  A = rand(3,3); A = A'*A;
  B = rand(2,3);
  C = rand(2,2); C = C'*C;
  K = [A B' ; B -C];

  # Dense Cholesky should throw an exception.
  @test_throws Compat.LinearAlgebra.PosDefException opCholesky(K);

  # Compute the LDL' factorization.
  LDL = opLDL(sparse(K));
  e = ones(size(K,1));
  @test(norm(LDL * (K * e) - e) < rtol * norm(e))

  # Test the Restriction and Extension index operators
  n = 10
  I = [1;2;4;7]
  r = 3:6
  s = 1:2:7
  k = 4
  v = rand(n)

  for idx in (I, r, s, Colon(), k)
    P = opRestriction(idx, n)
    Z = opExtension(idx, n)

    # 1d slices are different; in Julia, v[idx] is a scalar
    w = v[idx]
    typeof(idx) <: Number && (w = [w])
    vz = zeros(n); vz[idx] = v[idx]

    @test P * v == w
    @test P' * w == vz
    @test Z * w == vz
    @test Z' * v == w
    @test (P * Z) * w == w
    @test (Z * P) * v == vz
  end
end

test_linop()
