for A in Any[rand(2, 3), rand(-3:3, 2, 3),
             rand(2, 3) + im * rand(2, 3),
             sprand(10, 10, 0.1)]
  for B in Any[rand(2, 3), rand(-3:3, 2, 3),
               rand(2, 3) + im * rand(2, 3),
               sprand(10, 10, 0.1)]
    K = kron(A, B)
    normK = norm(K, 1)
    T1 = kron(LinearOperator(A), B)
    T2 = kron(A, LinearOperator(B))
    T3 = kron(LinearOperator(A), LinearOperator(B))
    for T in [T1, T2, T3]
      @test norm(K - full(T), 1) < eps() * normK
      @test norm(K' - full(T'), 1) < eps() * normK
      @test norm(K.' - full(T.'), 1) < eps() * normK
      m, n = size(K)
      err = 0.0
      for t = 1:100
        x = rand(n)
        Kx = K * x
        Tx = T * x
        err += norm(Kx - Tx)

        x = rand(m)
        Kx = K' * x
        Tx = T' * x
        err += norm(Kx - Tx)

        Kx = K.' * x
        Tx = T.' * x
        err += norm(Kx - Tx)
      end
      @test err < 1e-12
    end
  end
end
