# test hcat
A   = sprandn(100, 100, .5)
B   = randn(100, 10) + 1im * randn(100, 10)
C   = broadcast(x->round(Int64, x), randn(100, 90) * 30)
D   = [A B C]
Ao  = LinearOperator(A)
Bo  = LinearOperator(B)
Co  = LinearOperator(C)
Do  = [Ao Bo Co]
Do2 = LinearOperator(D)
rhs = randn(200)

@test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))
@test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))

@test_throws LinearOperatorException [LinearOperator(rand(5,5)) opEye(3)]

# test vcat
A   = sprandn(100, 100, 0.5)
B   = randn(10, 100) + 1im * randn(10, 100)
C   = broadcast(x->round(Int64, x), randn(90, 100) * 30)
D   = [A; B; C]
Ao  = LinearOperator(A)
Bo  = LinearOperator(B)
Co  = LinearOperator(C)
Do  = [Ao; Bo; Co]
Do2 = LinearOperator(D)
rhs = randn(100)

@test(norm(Do * rhs - D * rhs) <= rtol * norm(D * rhs))
@test(norm(Do2 * rhs - D * rhs) <= rtol * norm(D * rhs))

@test_throws LinearOperatorException [LinearOperator(rand(5,5)) ; opEye(3)]

