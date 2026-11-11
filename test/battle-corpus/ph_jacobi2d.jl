# ph_jacobi2d
# Jacobi iteration for -Laplace(u) = f on an (nint+2)-square grid, zero boundary, until the squared update falls below tol.
# LOOP i1 I K
# LOOP i2 I K
# LOOP i3 I K
# WHILE dd H
function ph_jacobi2d(u, v, f, h2, tol, nint, its)
    dd = 1.0
    k = 0
    while dd > tol
        # update into a second array: no in-place sweep, so every point is independent
        for i1 = 1:nint * nint
            r1 = div(i1 - 1, nint) + 2
            c1 = mod(i1 - 1, nint) + 2
            v[r1, c1] = 0.25 * (u[r1 - 1, c1] + u[r1 + 1, c1] + u[r1, c1 - 1] + u[r1, c1 + 1] + h2 * f[r1, c1])
        end
        # squared size of the update: a scalar reduction
        dd = 0.0
        for i2 = 1:nint * nint
            r2 = div(i2 - 1, nint) + 2
            c2 = mod(i2 - 1, nint) + 2
            dd = dd + (v[r2, c2] - u[r2, c2]) * (v[r2, c2] - u[r2, c2])
        end
        # copy back
        for i3 = 1:nint * nint
            r3 = div(i3 - 1, nint) + 2
            c3 = mod(i3 - 1, nint) + 2
            u[r3, c3] = v[r3, c3]
        end
        k = k + 1
    end
    its[1] = k
    return nothing
end
