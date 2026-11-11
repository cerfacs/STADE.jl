# ph_gs1d
# Gauss-Seidel iteration for -u'' = f on n points with zero boundary values, until the squared residual falls below tol. The sweep carries a value between iterations and must stay on the host.
# LOOP i1 S H
# LOOP i2 I K
# WHILE rs H
function ph_gs1d(u, f, h2, tol, n, its)
    rs = 1.0
    k = 0
    while rs > tol
        # in-place sweep: u[i-1] was written by the previous iteration
        for i1 = 2:n - 1
            u[i1] = 0.5 * (u[i1 - 1] + u[i1 + 1] + h2 * f[i1])
        end
        # residual of the discrete equation
        rs = 0.0
        for i2 = 2:n - 1
            rs = rs + (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2) * (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2)
        end
        k = k + 1
    end
    its[1] = k
    return nothing
end
