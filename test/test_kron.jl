function test_kron()
  @testset "Kron" begin
    for A in Any[simple_matrix(Float64, 2, 3),
                 simple_sparse_matrix(Float64, 10, 10)]
      for B in Any[simple_matrix(Float64, 2, 3),
                   simple_matrix(ComplexF64, 2, 3),
                   simple_sparse_matrix(Float64, 10, 10)]
        K = kron(A, B)
        normK = norm(K, 1)
        T1 = kron(LinearOperator(A), B)
        T2 = kron(A, LinearOperator(B))
        T3 = kron(LinearOperator(A), LinearOperator(B))
        for T in [T1, T2, T3]
          @test norm(K - Matrix(T), 1) < eps() * normK
          @test norm(K' - Matrix(T'), 1) < eps() * normK
          @test norm(transpose(K) - Matrix(transpose(T)), 1) < eps() * normK
          m, n = size(K)
          err = 0.0
          for t = 1:100
            x = simple_vector(Float64, n)
            Kx = K * x
            Tx = T * x
            err += norm(Kx - Tx)

            x = simple_vector(Float64, m)
            Kx = K' * x
            Tx = T' * x
            err += norm(Kx - Tx)

            Kx = transpose(K) * x
            Tx = transpose(T) * x
            err += norm(Kx - Tx)
          end
          @test err < 1e-12
        end
      end
    end
  end
end

test_kron()
