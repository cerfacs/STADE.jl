# lin_quotient(u, v, n, nsteps)
#
# The same ping-pong shape with quotients by a literal, and a literal tree as the divisor. Division by a literal is linear, so no value of u or v
# is needed by the adjoint.
#
# u: field, length n, input and final state
# v: scratch field, length n, overwritten
# n: number of points
# nsteps: number of (u -> v, v -> u) pairs of passes
function lin_quotient(u, v, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        # u -> v, interior points
        for i = 2:n - 1
            v[i] = (u[i - 1] + u[i + 1]) / 2.0 + u[i] / (2.0 * 2.0) - 0.5 * u[i] / 2.0
        end
        # v -> u, interior points
        for i = 2:n - 1
            u[i] = (v[i - 1] + v[i + 1]) / 2.0 + 0.5 * v[i]
        end
    end
    return nothing
end
