# br_gs(b, u, res, r0, n, maxit, tol)
#
# Gauss-Seidel sweeps for a 1D Poisson problem, with the residual norm boxed in length-1 arrays, at most maxit sweeps, each run while the norm is above 1.0e-12 * tol.
#
# b: right-hand side of length n
# u: solution of length n
# res: length-1 array for the squared residual
# r0: length-1 array for the residual norm
# n: number of points
# maxit: sweep cap
# tol: scale of the tolerance on the residual norm (1.0e-12 times it)
function br_gs(b, u, res, r0, n, maxit, tol)
    r0[1] = 1.0
    for k = 1:maxit
        if r0[1] > 1.0e-12 * tol
            # relaxation weight from the previous residual norm, so the boxed value reaches u
            w = 1.0 / (1.0 + r0[1])
            # in-place sweep: each point reads the point just updated
            for i = 2:n - 1
                u[i] = u[i] + w * (0.5 * (b[i] + u[i - 1] + u[i + 1]) - u[i])
            end
            res[1] = 0.0
            for i = 2:n - 1
                e = b[i] + u[i - 1] - 2.0 * u[i] + u[i + 1]
                res[1] += e * e
            end
            r0[1] = sqrt(res[1])
        end
    end
    return nothing
end
