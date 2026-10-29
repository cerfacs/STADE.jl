# lin_pingpong(u, v, n, nsteps)
#
# Two in-place diffusion passes per step, ping-pong between u and v, every coefficient a literal. Both arrays are overwritten each pass and no
# partial derivative needs a value of either, so the adjoint needs no snapshot of u or v.
#
# u: field, length n, input and final state
# v: scratch field, length n, overwritten
# n: number of points
# nsteps: number of (u -> v, v -> u) pairs of passes
function lin_pingpong(u, v, n, nsteps)
    # boundary values are fixed in both buffers
    v[1] = u[1]
    v[n] = u[n]
    # the pass loop carries the field from one pass to the next
    for t = 1:nsteps
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
