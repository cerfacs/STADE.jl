# lin_quad_inside(u, v, loss, n, nsteps)
#
# The same passes with a quadratic term read inside the pass loop, before the next overwrite of u. Here the old value of u is needed by that
# term, so the adjoint must keep the snapshot of u, while the snapshot of v stays unnecessary.
#
# u: field, length n, input and final state
# v: scratch field, length n, overwritten
# loss: length-1 output array, accumulated in place
# n: number of points
# nsteps: number of (u -> v, v -> u) pairs of passes
function lin_quad_inside(u, v, loss, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        # nonlinear read of u before it is overwritten below
        for i = 1:n
            loss[1] = loss[1] + 0.5 * u[i] * u[i]
        end
        # u -> v, interior points
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        end
        # v -> u, interior points
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * (v[i - 1] - 2.0 * v[i] + v[i + 1])
        end
    end
    return nothing
end
