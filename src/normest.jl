"""
  normest(S) estimates the matrix 2-norm of S.

  This method has been adopted from the MATLAB counterpart
  This function is also a minor adaptation of Matlab's built-in NORMEST.
  This method allocates.

  -----------------------------------------
  Inputs:
    S --- Matrix or LinearOperator type, 
    tol ---  relative error tol, default 1.0e-6
    maxiter --- maximum iteration, default 100
    
  Returns:
    e --- the estimated norm
    cnt --- the number of iterations used
  """
function normest(S, tol = 1.0e-6, maxiter = 100)
    (m, n) = size(S)
    cnt = 0

    # Compute an "estimate" of the ab-val column sums.
    v = ones(eltype(S), m)
    v[randn(m).<0] .= -1
    x = abs.(S' * v)
    e = norm(x)

    if e == 0
        return e, cnt
    end

    x = x / e
    e_0 = zero(e)

    while abs(e - e_0) > tol * e
        e_0 = e
        Sx = S * x
        if count(x -> x != 0, Sx) == 0
            Sx = randn(eltype(Sx), size(Sx))
        end
        x = S' * Sx
        normx = norm(x)
        e = normx / norm(Sx)
        x = x / normx
        cnt = cnt + 1
        if cnt > maxiter
            error(
                "normest did not converge for ",
                maxiter,
                "  iterations with tolerance ",
                tol,
            )
            break
        end
    end

    return e, cnt
end
