using Base.Test
using linop

(nrow, ncol) = (10, 6);
rtol = sqrt(eps(Float64));

# Test LinearOperator(Matrix).
A1 = rand(nrow, ncol) + rand(nrow, ncol) * im;
A2 = sprand(nrow, ncol, 0.5) + sprand(nrow, ncol, 0.5) * im;

for A in (A1, A2)
  op = LinearOperator(A);
  @test(op.nrow == nrow);
  @test(op.ncol == ncol);

  @test(vecnorm(A   - full(op))   <= rtol * vecnorm(A));
  @test(vecnorm(A.' - full(op.')) <= rtol * vecnorm(A));
  @test(vecnorm(A'  - full(op'))  <= rtol * vecnorm(A));

  v = rand(ncol) + rand(ncol) * im;
  @test(norm(A * v - op * v) <= rtol * norm(v));

  u = rand(nrow) + rand(nrow) * im;
  @test(norm(A.' * u - op.' * u) <= rtol * norm(u));
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
  @test(norm(opC.' * u - C.' * u) <= rtol * norm(u));
  @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));
end

B2 = rand(ncol, ncol+1) + rand(ncol, ncol+1) * im;
C = A1 * B2;
opC = LinearOperator(A1) * LinearOperator(B2);
v = rand(ncol+1) + rand(ncol+1) * im;
@test(norm(opC * v - C * v) <= rtol * norm(v));
u = rand(nrow) + rand(nrow) * im;
@test(norm(opC.' * u - C.' * u) <= rtol * norm(u));
@test(norm(opC'  * u - C'  * u) <= rtol * norm(u));

# Test Cholesky operator.
AA = A1' * A1;
AAinv = opCholesky(AA, check=true);
v = rand(ncol) + im * rand(ncol);
@test(norm(AAinv * v - AA \ v) <= rtol * norm(v));

# Test Hermitian operator.
A1 = rand(nrow, nrow) + rand(nrow, nrow) * im;
d = diag(A1); A1 = tril(A1, -1);
A2 = A1 + A1' + diagm(d);
op = opHermitian(d, A1);
v  = rand(nrow) + rand(nrow) * im;
Av = A2 * v;
@test(norm(op * v - Av) <= rtol * norm(Av));
