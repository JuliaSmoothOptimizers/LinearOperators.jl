using LinearOperators
using Base.Test

ϵ = eps(Float64);
rtol = sqrt(ϵ);

# test hcat
A = sprandn(100,100,0.1)
B = sprandn(100,10,0.6)
C = [A B]
Ao = LinearOperator(A)
Bo = LinearOperator(B)
Co = [Ao Bo]
Co2 = LinearOperator(C)
rhs = randn(110)
@test(norm(Co*rhs - C*rhs) <= rtol * norm(C*rhs));
@test(norm(Co2*rhs - C*rhs) <= rtol * norm(C*rhs));

# test vcat
A = sprandn(100,100,0.1)
B = sprandn(10,100,0.6)
C = [A; B]
Ao = LinearOperator(A)
Bo = LinearOperator(B)
Co = [Ao; Bo]
Co2 = LinearOperator(C)
rhs = randn(100)
@test(norm(Co*rhs - C*rhs) <= rtol * norm(C*rhs));
@test(norm(Co2*rhs - C*rhs) <= rtol * norm(C*rhs));

# test hcat
A = randn(100,100)
B = randn(100,10)
C = [A B]
Ao = LinearOperator(A)
Bo = LinearOperator(B)
Co = [Ao Bo]
Co2 = LinearOperator(C)
rhs = randn(110)
@test(norm(Co*rhs - C*rhs) <= rtol * norm(C*rhs));
@test(norm(Co2*rhs - C*rhs) <= rtol * norm(C*rhs));

# test vcat
A = randn(100,100)
B = randn(10,100)
C = [A; B;]
Ao = LinearOperator(A)
Bo = LinearOperator(B)
Co = [Ao; Bo;]
Co2 = LinearOperator(C)
rhs = randn(100)
@test(norm(Co*rhs - C*rhs) <= rtol * norm(C*rhs));
@test(norm(Co2*rhs - C*rhs) <= rtol * norm(C*rhs));