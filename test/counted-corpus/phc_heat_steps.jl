# phc_heat_steps
# Backward-Euler heat equation for nsteps time steps. Each step solves (I - lam L) w = u by Jacobi iteration, at most maxit sweeps, each run only while the squared update is above tol (a counted loop with a guard replaces the while). The time loop carries u from step to step. The trip count is not reported: a counter stored in an output array is a differentiable output, and keep_push_pop = false refuses the integer it restores.
# LOOP t1 S H
# LOOP i1 I K
# LOOP i2 I K
# LOOP i3 I K
# LOOP i4 I K
# LOOP i5 I K
function phc_heat_steps(u, w, wn, lam, tol, n, nsteps, maxit)
    for t1 = 1:nsteps
        # initial guess: the previous state
        for i1 = 1:n
            w[i1] = u[i1]
        end
        dd = 1.0
        for kk = 1:maxit
            if dd > tol
                # Jacobi update, interior points; the boundary values stay fixed
                wn[1] = u[1]
                wn[n] = u[n]
                for i2 = 2:n - 1
                    wn[i2] = (u[i2] + lam * (w[i2 - 1] + w[i2 + 1])) / (1.0 + 2.0 * lam)
                end
                dd = 0.0
                for i3 = 1:n
                    dd = dd + (wn[i3] - w[i3]) * (wn[i3] - w[i3])
                end
                for i5 = 1:n
                    w[i5] = wn[i5]
                end
            end
        end
        for i4 = 1:n
            u[i4] = w[i4]
        end
    end
    return nothing
end
