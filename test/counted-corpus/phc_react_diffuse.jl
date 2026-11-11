# phc_react_diffuse
# One operator-split step: a backward-Euler cubic decay solved per cell by Newton's method (counted loop with a guard, at most maxit steps), then an explicit diffusion stencil.
# LOOP i1 I K
# LOOP i2 I K
function phc_react_diffuse(u0, v, u, kap, dt, nu, tol, n, maxit)
    # reaction: v + dt kap v^3 = u0, per cell
    for i1 = 1:n
        v1 = u0[i1]
        st = 1.0
        for k1 = 1:maxit
            if st > tol
                dlt = (v1 + dt * kap * v1 * v1 * v1 - u0[i1]) / (1.0 + 3.0 * dt * kap * v1 * v1)
                v1 = v1 - dlt
                st = abs(dlt)
            end
        end
        v[i1] = v1
    end
    # diffusion: interior stencil, boundary copied through
    u[1] = v[1]
    u[n] = v[n]
    for i2 = 2:n - 1
        u[i2] = v[i2] + nu * (v[i2 - 1] - 2.0 * v[i2] + v[i2 + 1])
    end
    return nothing
end
