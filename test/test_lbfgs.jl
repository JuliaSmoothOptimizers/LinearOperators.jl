ϵ = eps(Float64)
rtol = sqrt(ϵ)

# test limited-memory BFGS
n = 10
mem = 5
B = LBFGSOperator(n, mem)
H = InverseLBFGSOperator(n, mem)

for t = 1:2 # Run again after reset!
  @assert norm(diag(B) - diag(full(B))) <= rtol

  @assert B.data.insert == 1
  @assert H.data.insert == 1
  @test norm(full(B) - eye(n)) <= ϵ
  @test norm(full(H) - eye(n)) <= ϵ

  # Test that nonpositive curvature can't be added.
  s = rand(n)
  z = zeros(n)
  push!(B, s, -s); @assert B.data.insert == 1
  push!(B, s,  z); @assert B.data.insert == 1
  push!(H, s, -s); @assert H.data.insert == 1
  push!(H, s,  z); @assert H.data.insert == 1

  # Insert a few {s,y} pairs.
  for i = 1 : mem+2
    s = rand(n)
    y = rand(n)
    if dot(s, y) >= 1.0e-20
      push!(B, s, y)
      push!(H, s, y)
    end
  end

  @assert B.data.insert == 3
  @assert H.data.insert == 3

  @test check_positive_definite(B)
  @test check_positive_definite(H)

  @test check_hermitian(B)
  @test check_hermitian(H)

  @assert norm(diag(B) - diag(full(B))) <= rtol

  @test norm(full(H * B) - eye(n)) <= rtol

  # Testing reset! function
  v = rand(n)
  @test norm(B * v - v) > rtol
  @test norm(H * v - v) > rtol
  reset!(B)
  reset!(H)
  @test norm(B * v - v) < rtol
  @test norm(H * v - v) < rtol
end

# test against full BFGS without scaling
mem = n
LB = LBFGSOperator(n, mem)
B = eye(n)

function bfgs!(B, s, y, damped=false)
  # dense BFGS update
  ys = dot(y, s)
  Bs = B * s
  tol = damped ? (0.2 * dot(s, Bs)) : 1.0e-20
  if ys > tol
    B = B - Bs * Bs' / dot(s, Bs) + y * y' / ys
  end
  return B
end

@assert norm(full(LB) - B) < rtol * norm(B)
@assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

for k = 1 : mem
  s = rand(n)
  y = rand(n)
  B = bfgs!(B, s, y)
  LB = push!(LB, s, y)
  @assert norm(full(LB) - B) < rtol * norm(B)
  @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
end

# test damped L-BFGS
B = LBFGSOperator(n, mem, damped=true)
H = InverseLBFGSOperator(n, mem, damped=true)

insert_B = insert_H = 1
for i = 1 : mem+2
  s = rand(n)
  y = rand(n)
  ys = dot(y, s)
  if ys > B.data.damp_factor * dot(s, B * s)
    insert_B += 1
    push!(B, s, y)
  end
  if ys > B.data.damp_factor * dot(y, H * y)
    insert_H += 1
    push!(H, s, y)
  end
end

@assert B.data.insert == mod(insert_B, B.data.mem)
@assert H.data.insert == mod(insert_H, H.data.mem)

@test check_positive_definite(B)
@test check_positive_definite(H)

@test check_hermitian(B)
@test check_hermitian(H)

@assert norm(diag(B) - diag(full(B))) <= rtol

@test norm(full(H * B) - eye(n)) <= rtol

# test against full BFGS without scaling
mem = n
LB = LBFGSOperator(n, mem, damped=true)
B = eye(n)

@assert norm(full(LB) - B) < rtol * norm(B)
@assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))

for k = 1 : mem
  s = rand(n)
  y = rand(n)
  B = bfgs!(B, s, y, true)
  LB = push!(LB, s, y)
  @assert norm(full(LB) - B) < rtol * norm(B)
  @assert norm(diag(LB) - diag(B)) < rtol * norm(diag(B))
end
