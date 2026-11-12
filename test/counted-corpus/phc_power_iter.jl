# phc_power_iter
# Dominant eigenpair of the tridiagonal operator (A x)[i] = c[i] x[i] - x[i-1] - x[i+1], by power iteration with a Rayleigh quotient, at most maxit steps, each run only while the eigenvalue estimate changed by more than tol. The trip count is not reported: a counter stored in an output array is a differentiable output, and keep_push_pop = false refuses the integer it restores.
# LOOP i1 I K
# LOOP i2 I K
# LOOP i3 I K
# LOOP i4 I K
function phc_power_iter(c, x, y, lam, tol, n, maxit)
    dl = 1.0
    old = 0.0
    for kk = 1:maxit
        if dl > tol
            # operator: boundary rows then the interior
            y[1] = c[1] * x[1] - x[2]
            y[n] = c[n] * x[n] - x[n - 1]
            for i1 = 2:n - 1
                y[i1] = c[i1] * x[i1] - x[i1 - 1] - x[i1 + 1]
            end
            # Rayleigh quotient (x is normalised) and the norm of y
            s1 = 0.0
            for i2 = 1:n
                s1 = s1 + x[i2] * y[i2]
            end
            s2 = 0.0
            for i3 = 1:n
                s2 = s2 + y[i3] * y[i3]
            end
            nrm = sqrt(s2)
            for i4 = 1:n
                x[i4] = y[i4] / nrm
            end
            dl = abs(s1 - old)
            old = s1
        end
    end
    lam[1] = old
    return nothing
end
