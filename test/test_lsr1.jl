function test_lsr1()
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)

  # test limited-memory SR1
  n = 10
  mem = 5
  B = LSR1Operator(n, mem, scaling=false)

  for t = 1:2
    @assert norm(diag(B) - diag(full(B))) <= rtol

    @assert B.data.insert == 1
    @test norm(full(B) - Matrix(1.0I, n, n)) <= ϵ

    # Test that only valid updates are accepted.
    s = rand(n)
    y = B * s
    push!(B, s, y); @assert B.data.insert == 1

    # Insert a few {s,y} pairs.
    for i = 1 : mem+2
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
    end

    @test check_hermitian(B)
    @assert norm(diag(B) - diag(full(B))) <= rtol

    v = rand(n)
    @assert norm(B * v - v) > rtol
    reset!(B)
    @assert norm(B * v - v) < rtol
  end

  # test against full SR1 without scaling
  mem = n
  LB = LSR1Operator(n, mem, scaling=false)
  B = Matrix(1.0I, n, n)

  function sr1!(B, s, y)
    # dense SR1 update
    ymBs = y - B * s
    denom = dot(ymBs, s)
    if abs(denom) >= 1.0e-8 + 1.0e-8 * norm(s) * norm(ymBs)
      B = B + ymBs * ymBs' / denom
    end
    return B
  end

  @assert norm(full(LB) - B) < rtol * norm(B)
  @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

  for k = 1 : mem
    s = rand(n)
    y = rand(n)
    B = sr1!(B, s, y)
    LB = push!(LB, s, y)
    @assert norm(full(LB) - B) < rtol * norm(B)
    @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
  end
end

test_lsr1()
