using Zygote

function matmulOp(mat::AbstractArray{T}) where {T}
  function prod!(res, x)
    for i in axes(mat, 1)
      res[i] = transpose(mat[i, :]) * x
    end
  end

  function ctprod!(res, x)
    for i in axes(mat, 2)
      res[i] = dot(mat[:, i], x)
    end
  end

  return LinearOperator(T, size(mat, 1), size(mat, 2), false, false, prod!, nothing, ctprod!)
end

function test_chainrules()
  @testset ExtendedTestSet "Chainrules" begin
    for (M, N) in zip([2, 3, 8, 7], [2, 4, 8, 16])
      for T in [Float64, ComplexF64]
        mat = simple_matrix(T, M, N)
        op = matmulOp(mat)
        x = rand(T, N)
        xᵀ = transpose(x[1:M])
        xᴴ = adjoint(x[1:M])

        # test op*x
        y, g = Zygote.withgradient(v -> sum(abs.(op * v)), x)
        y2, g2 = Zygote.withgradient(v -> sum(abs.(mat * v)), x)
        @test isapprox(y, y2)
        @test isapprox(g[1], g2[1])

        # test xᵀ*op
        yt, gt = Zygote.withgradient(v -> sum(abs.(v * op)), xᵀ)
        yt2, gt2 = Zygote.withgradient(v -> sum(abs.(v * mat)), xᵀ)
        @test isapprox(yt, yt2)
        @test isapprox(gt[1], gt2[1])

        # test xᴴ*op
        yh, gh = Zygote.withgradient(v -> sum(abs.(v * op)), xᴴ)
        yh2, gh2 = Zygote.withgradient(v -> sum(abs.(v * mat)), xᴴ)
        @test isapprox(yh, yh2)
        @test isapprox(gh[1], gh2[1])
      end
    end
  end
end

test_chainrules()
