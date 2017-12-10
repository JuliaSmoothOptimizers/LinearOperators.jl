for A in Any[rand(2, 3), rand(-3:3, 2, 3),
             rand(2, 3) + im * rand(2, 3)]
  for B in Any[rand(2, 3), rand(-3:3, 2, 3),
               rand(2, 3) + im * rand(2, 3)]
    K = kron(A, B)
    T = opKron(A, B)
    @test K == full(T)
    @test K' == full(T')
    @test K.' == full(T.')
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

A = rand(5, 5)
B = rand(5, 5)
TB = LinearOperator(B)
K = kron(A, B)
T = opKron(A, TB)
@test K == full(T)
@test K' == full(T')
@test K.' == full(T.')
