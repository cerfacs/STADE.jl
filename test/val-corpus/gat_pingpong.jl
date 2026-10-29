# gat_pingpong(u, v, n, nsteps)
#
# Two explicit three-point steps per pass, ping-pong between u and v (the shape of a time loop around two stencil loops). The adjoint of each
# inner loop scatters into the other array at offsets -1, 0, +1.
#
# u: field, length n, input and final state
# v: scratch field, length n, overwritten
# n: number of points
# nsteps: number of passes
function gat_pingpong(u, v, n, nsteps)
    # fixed boundary values in both buffers
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        # u -> v, interior
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        end
        # v -> u, interior
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * (v[i - 1] - 2.0 * v[i] + v[i + 1])
        end
    end
    return nothing
end
