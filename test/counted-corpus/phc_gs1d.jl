# phc_gs1d
# Gauss-Seidel iteration for -u'' = f on n points with zero boundary values, at most maxit sweeps, each run only while the squared residual is above tol. The sweep carries a value between iterations and must stay on the host. The trip count is not reported: a counter stored in an output array is a differentiable output, and keep_push_pop = false refuses the integer it restores.
# LOOP i1 S H
# LOOP i2 I K
function phc_gs1d(u, f, h2, tol, n, maxit)
    rs = 1.0
    for kk = 1:maxit
        if rs > tol
            # in-place sweep: u[i-1] was written by the previous iteration
            for i1 = 2:n - 1
                u[i1] = 0.5 * (u[i1 - 1] + u[i1 + 1] + h2 * f[i1])
            end
            # residual of the discrete equation
            rs = 0.0
            for i2 = 2:n - 1
                rs = rs + (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2) * (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2)
            end
        end
    end
    return nothing
end
