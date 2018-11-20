using LinearAlgebra, SparseArrays

function test_linop()
  (nrow, ncol) = (10, 6);
  ϵ = eps(Float64);
  rtol = sqrt(ϵ);
  A1 = rand(nrow, ncol) + rand(nrow, ncol) * im;

  @testset "Basic operations" begin
    for op = (LinearOperator(A1), PreallocatedLinearOperator(A1))
      show(op);

      @testset "Data type" begin
        @test eltype(op) == eltype(A1)
        @test !isreal(op)
      end

      @testset "Size" begin
        @test(size(op) == (nrow, ncol));
        @test(shape(op) == (nrow, ncol));
        @test(size(op, 1) == nrow);
        @test(size(op, 2) == ncol);
        @test_throws LinearOperatorException size(op, 3)
        @test_throws LinearOperatorException op * rand(ncol + 1)
      end

      @testset "Boolean operators" begin
        @test(symmetric(op) == false);
        @test(hermitian(op) == false);
      end

      @testset "Full" begin
        @test(norm(A1 - Matrix(op)) <= ϵ * norm(A1));
      end

      @testset "Unary +." begin
        @test(norm(A1 - Matrix(+op)) <= ϵ * norm(A1));
      end
    end

    @testset "LinearOperator(Matrix)" begin
      A2 = sprand(nrow, ncol, 0.5) + sprand(nrow, ncol, 0.5) * im;
      for A in (A1, A2)
        op = LinearOperator(A);
        @test(op.nrow == nrow);
        @test(op.ncol == ncol);

        @test(norm(A   - Matrix(op))   <= rtol * norm(A));
        @test(norm(transpose(A) - Matrix(transpose(op))) <= rtol * norm(A));
        @test(norm(A'  - Matrix(op'))  <= rtol * norm(A));

        v = rand(ncol) + rand(ncol) * im;
        @test(norm(A * v - op * v) <= rtol * norm(v));

        u = rand(nrow) + rand(nrow) * im;
        @test(norm(transpose(A) * u - transpose(op) * u) <= rtol * norm(u));
        @test(norm(A'  * u - op'  * u) <= rtol * norm(u));
      end
    end

    @testset "Preallocated LinearOperator(Matrix)" begin
      A2 = rand(nrow, ncol)
      for A in (A1, A2)
        T = eltype(A)
        Mv = zeros(T, nrow)
        Mtu = zeros(T, ncol)
        Maw = zeros(T, ncol)
        op = PreallocatedLinearOperator(Mv, Mtu, Maw, A)
        v = T <: Real ? rand(ncol) : rand(ncol) + rand(ncol) * im;
        u = T <: Real ? rand(nrow) : rand(nrow) + rand(nrow) * im;
        @test norm(op * v - A * v) <= rtol * norm(A)
        @test norm(transpose(op) * u - transpose(A) * u) <= rtol * norm(A)
        @test norm(op' * u - A' * u) <= rtol * norm(A)

        al = @allocated op * v
        @test al == 0
        al = @allocated adjoint(op) * u
        @test al == 10 * Int.size
        al = @allocated op' * u
        @test al == 10 * Int.size

        op = PreallocatedLinearOperator(A)
        v = T <: Real ? rand(ncol) : rand(ncol) + rand(ncol) * im;
        u = T <: Real ? rand(nrow) : rand(nrow) + rand(nrow) * im;
        @test norm(op * v - A * v) <= rtol * norm(A)
        @test norm(transpose(op) * u - transpose(A) * u) <= rtol * norm(A)
        @test norm(op' * u - A' * u) <= rtol * norm(A)
      end

      for B in (rand(nrow, nrow), sprand(nrow, nrow, 0.5))
        for A in (SymTridiagonal(Symmetric(B)), Symmetric(B))
          v = rand(nrow)

          Mv = zeros(nrow)
          op = PreallocatedLinearOperator(Mv, A)
          @test norm(op * v - A * v) <= rtol * norm(A)
          @test norm(transpose(op) * v - transpose(A) * v) <= rtol * norm(A)
          @test norm(op' * v - A' * v) <= rtol * norm(A)

          op = PreallocatedLinearOperator(A)
          @test norm(op * v - A * v) <= rtol * norm(A)
          @test norm(transpose(op) * v - transpose(A) * v) <= rtol * norm(A)
          @test norm(op' * v - A' * v) <= rtol * norm(A)
        end
      end
    end

    @testset "Basic arithmetic operations" begin
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
    end

    B2 = rand(ncol, ncol+1) + rand(ncol, ncol+1) * im;
    @testset "Operator ± scalar" begin
      opC = LinearOperator(A1) + 2.12345;
      @test(norm(A1 .+ 2.12345 - Matrix(opC)) <= rtol * norm(A1 .+ 2.12345));

      opC = 2.12345 + LinearOperator(A1);
      @test(norm(A1 .+ 2.12345 - Matrix(opC)) <= rtol * norm(A1 .+ 2.12345));

      opC = LinearOperator(A1) - 2.12345;
      @test(norm((A1 .- 2.12345) - Matrix(opC)) <= rtol * norm(A1 .- 2.12345));

      opC = 2.12345 - LinearOperator(A1);
      @test(norm((2.12345 .- A1) - Matrix(opC)) <= rtol * norm(2.12345 .- A1));

      C = A1 * B2;
      opC = LinearOperator(A1) * LinearOperator(B2);
      v = rand(ncol+1) + rand(ncol+1) * im;
      @test(norm(opC * v - C * v) <= rtol * norm(v));
      u = rand(nrow) + rand(nrow) * im;
      @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u));
      @test(norm(opC'  * u - C'  * u) <= rtol * norm(u));

      @test_throws LinearOperatorException LinearOperator(A1) + LinearOperator(B2);
      @test_throws LinearOperatorException LinearOperator(B2) * LinearOperator(A1);
    end

    A1B2 = A1 * B2;
    @testset "Matrix × operator" begin
      opC = A1 * LinearOperator(B2);
      @test(norm(A1B2 - Matrix(opC)) <= rtol * norm(A1B2));
    end

    @testset "Operator × matrix" begin
      opC = LinearOperator(A1) * B2;
      @test(norm(A1B2 - Matrix(opC)) <= rtol * norm(A1B2));
    end

    AA1 = 2.12345 * A1;
    @testset "Scalar × operator" begin
      opC = 2.12345 * LinearOperator(A1);
      @test(norm(AA1 - Matrix(opC)) <= rtol * norm(AA1));
    end

    @testset "Operator × scalar" begin
      opC = LinearOperator(A1) * 2.12345;
      @test(norm(AA1 - Matrix(opC)) <= rtol * norm(AA1));
    end
  end

  @testset "Basic operators" begin
    @testset "Identity" begin
      opI = opEye(nrow);
      v = rand(nrow) + rand(nrow) * im;
      @test(abs(norm(opI * v - v)) <= ϵ * norm(v));
      @test(abs(norm(transpose(opI) * v - v)) <= ϵ * norm(v));
      @test(abs(norm(opI' * v - v)) <= ϵ * norm(v));
      @test(norm(Matrix(opI) - Matrix(1.0I, nrow, nrow)) <= ϵ * norm(Matrix(1.0I, nrow, nrow)));

      w = opI * v
      w[1] = -1.0
      @test v[1] != w[1]

      opI = opEye(nrow, ncol)
      v = rand(ncol) + rand(ncol) * im
      v0 = [v ; zeros(nrow - ncol)]
      vu = [v ; rand(nrow - ncol)]
      @test(abs(norm(opI * v - v0)) <= ϵ * norm(v))
      @test(abs(norm(transpose(opI) * vu - v)) <= ϵ * norm(v))
      @test(abs(norm(opI' * vu - v)) <= ϵ * norm(v))
      @test(norm(Matrix(opI) - Matrix(1.0I, nrow, ncol)) <= ϵ * norm(Matrix(1.0I, nrow, ncol)))

      opI = opEye(ncol, nrow)
      @test(abs(norm(opI * vu - v)) <= ϵ * norm(v))
      @test(abs(norm(transpose(opI) * v - v0)) <= ϵ * norm(v))
      @test(abs(norm(opI' * v - v0)) <= ϵ * norm(v))
      @test(norm(Matrix(opI) - Matrix(1.0I, ncol, nrow)) <= ϵ * norm(Matrix(1.0I, ncol, nrow)))
    end

    @testset "Identity (non-convertible to matrix)" begin
      op = opEye()

      v = rand(5)
      w = op * v
      @test w === v
      w = v * op

      @test w === v
      A2 = op * A1
      @test A2 === A1
      A2 = A1 * op
      @test A2 === A1

      T1 = LinearOperator(A1)
      T2 = op * T1
      @test T2 === T1
      T2 = T1 * op
      @test T2 === T1

      op2 = opEye()
      @test op === op2
      @test op === op * op2 === op2 * op
    end

    @testset "Ones" begin
      E = opOnes(nrow, ncol);
      v = rand(nrow) + rand(nrow) * im;
      u = rand(ncol) + rand(ncol) * im;
      @test(norm(E * u - sum(u) * ones(nrow)) <= rtol * norm(u));
      @test(norm(transpose(E) * v - sum(v) * ones(ncol)) <= rtol * norm(v));
      @test(norm(E' * v - sum(v) * ones(ncol)) <= rtol * norm(v));
    end

    @testset "Zeros" begin
      O = opZeros(nrow, ncol);
      v = rand(nrow) + rand(nrow) * im;
      u = rand(ncol) + rand(ncol) * im;
      @test(norm(O * u) <= ϵ);
      @test(norm(transpose(O) * v) <= ϵ);
      @test(norm(O' * v) <= ϵ);
    end

    @testset "Diagonal" begin
      v = rand(nrow) + rand(nrow) * im;
      D = opDiagonal(v);
      u = rand(nrow) + rand(nrow) * im;
      @test(norm(D * u - v .* u) <= ϵ * norm(u));
      @test(norm(transpose(D) * u - v .* u) <= ϵ * norm(u));
      @test(norm(D' * u - conj(v) .* u) <= ϵ * norm(u));
    end

    @testset "Rectangular diagonal" begin
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
    end

    @testset "Hermitian" begin
      A = rand(nrow, nrow) + rand(nrow, nrow) * im;
      d = real.(diag(A)); A = tril(A, -1);
      C = A + A' + diagm(0 => d)
      H = opHermitian(d, A);
      v = rand(nrow) + rand(nrow) * im;
      @test(norm(H * v - C * v) <= rtol * norm(v));
      @test(norm(transpose(H) * v - transpose(C) * v) <= rtol * norm(v));
      @test(norm(H' * v - C * v) <= rtol * norm(v));

      @test(! check_hermitian(LinearOperator(A - A')));
      @test(! check_positive_definite(LinearOperator(-A'*A)));

      A = rand(nrow, nrow) + rand(nrow, nrow) * im;
      C = A + A'
      H = opHermitian(C);
      v = rand(nrow) + rand(nrow) * im;
      @test(norm(H * v - C * v) <= rtol * norm(v));
      @test(norm(transpose(H) * v - transpose(C) * v) <= rtol * norm(v));
      @test(norm(H' * v - C * v) <= rtol * norm(v));
    end

    @testset "Transpose and adjoint" begin
      A = rand(nrow, nrow) + rand(nrow, nrow) * im;
      v = rand(nrow) + rand(nrow) * im;

      op = LinearOperator(nrow, nrow, false, false, v->A*v, v->transpose(A)*v, nothing)
      @test(norm(transpose(A) * v - transpose(op) * v) <= rtol * norm(v))
      @test(norm(adjoint(A) * v - adjoint(op) * v) <= rtol * norm(v))
      @test(norm(A * v - transpose(transpose(op)) * v) <= rtol * norm(v))
      @test(norm(A * v - adjoint(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - transpose(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - adjoint(transpose(op)) * v) <= rtol * norm(v))

      op = LinearOperator(nrow, nrow, false, false, v->A*v, nothing, v->adjoint(A)*v)
      @test(norm(transpose(A) * v - transpose(op) * v) <= rtol * norm(v))
      @test(norm(adjoint(A) * v - adjoint(op) * v) <= rtol * norm(v))
      @test(norm(A * v - transpose(transpose(op)) * v) <= rtol * norm(v))
      @test(norm(A * v - adjoint(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - transpose(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - adjoint(transpose(op)) * v) <= rtol * norm(v))
    end

    @testset "Integer" begin
        A = convert(Array{Int,2}, round.(10 * rand(nrow, nrow) .- 5))
        op = LinearOperator(A)
        @test check_ctranspose(op)
        @test check_hermitian(op + op')
        @test check_positive_definite(op * op')
    end

    @testset "Restriction and Extension" begin
      n = 10
      J = [1;2;4;7]
      r = 3:6
      s = 1:2:7
      k = 4
      v = rand(n)

      for idx in (J, r, s, Colon(), k)
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
  end

  @testset "Linear system operators" begin
    (U, _) = qr(rand(nrow, nrow) + rand(nrow, nrow) * im);
    (V, _) = qr(rand(nrow, nrow) + rand(nrow, nrow) * im);
    Σ = diagm(0 => rand(nrow) .+ 0.1);
    A = U * Σ * V';
    v = rand(nrow) + rand(nrow) * im;

    @testset "Inverse" begin
      Ainv = opInverse(A);
      @test(norm(A \ v - Ainv * v) <= rtol * norm(v));
      @test(norm(transpose(A) \ v - transpose(Ainv) * v) <= rtol * norm(v));
      @test(norm(A' \ v - Ainv' * v) <= rtol * norm(v));
    end

    @testset "Cholesky and LDL" begin
      B = A' * A;
      Binv = opCholesky(B)  #, check=true);
      @test(norm(B \ v - Binv * v) <= rtol * norm(v));
      @test(norm(transpose(B) \ v - transpose(Binv) * v) <= rtol * norm(v));
      @test(norm(B' \ v - Binv' * v) <= rtol * norm(v));

      @test_throws LinearOperatorException opCholesky(rand(3,5));
      @test_throws LinearOperatorException opCholesky(rand(5,5), check=true);

      # Test Cholesky operator on SQD matrix.
      A = rand(3,3); A = A'*A;
      B = rand(2,3);
      C = rand(2,2); C = C'*C;
      K = [A B' ; B -C];

      # Dense Cholesky should throw an exception.
      @test_throws LinearAlgebra.PosDefException opCholesky(K);

      # Compute the LDL' factorization.
      LDL = opLDL(sparse(K));
      e = ones(size(K,1));
      @test(norm(LDL * (K * e) - e) < rtol * norm(e))
    end

    @testset "Householder" begin
      v = rand(nrow) + rand(nrow) * im;
      H = opHouseholder(v);
      u = rand(nrow) + rand(nrow) * im;
      @test(norm(H * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u));
      @test(norm(transpose(H) * u - (u - 2 * dot(conj(v), u) * conj(v))) <= rtol * norm(u));
      @test(norm(H' * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u));
    end
  end

  @testset "Inference" begin
    op = LinearOperator(5, 3, false, false,
                        p -> ones(5) + im * ones(5));
    @test eltype(op) == ComplexF64
    @test_throws LinearOperatorException transpose(op)  # cannot be inferred
    @test_throws LinearOperatorException op'            # cannot be inferred

    op2 = conj(op);
    @test(norm(Matrix(op2) - conj(Matrix(op))) <= ϵ * norm(Matrix(op)));

    A = rand(5,3) + im * rand(5,3);
    op = LinearOperator(A);
    @test(check_ctranspose(A));
    @test(check_ctranspose(op));
    @test_throws LinearOperatorException opCholesky(A)  # Shape mismatch

    A = rand(5,5) + im * rand(5,5);
    @test_throws LinearOperatorException opCholesky(A, check=true)  # Not Hermitian / positive definite
    @test_throws LinearOperatorException opCholesky(-A'*A, check=true)  # Not positive definite
  end

  @testset "Type specific operator" begin
    prod = v -> [v[1] + v[2]; v[2]]
    ctprod = v -> [v[1]; v[1] + v[2]]
    op = LinearOperator(2, 2, false, false, prod, nothing, ctprod)
    @test eltype(op) == Complex{Float64}
    for T in (Complex{Float64}, Complex{Float32}, Float64, Float32, Int32)
      op = LinearOperator(T, 2, 2, false, false, prod, nothing, ctprod)
      @test eltype(op) == T
    end

    A = [im 1.0; 0.0 1.0]
    prod = v -> A * v
    tprod = u -> transpose(A) * u
    ctprod = w -> A' * w
    opC = LinearOperator(2, 2, false, false, prod, tprod, ctprod)
    v = rand(2) + rand(2) * im
    @test A == Matrix(opC)
    opF = LinearOperator(Float64, 2, 2, false, false, prod, tprod, ctprod) # The type is a lie
    @test eltype(opF) == Float64
    @test_throws InexactError Matrix(opF)
  end

  # Issue #80
  @testset "Test mul!" begin
    A = [1.0 1.0; 1.0 0.0]
    op = LinearOperator(A)
    y = zeros(2)
    x = ones(2)
    mul!(y, op, x)
    @test y == [2.0; 1.0]
  end
end

test_linop()
