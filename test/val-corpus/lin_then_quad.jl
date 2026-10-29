# lin_then_quad(u, v, loss, n, nsteps)
#
# Linear ping-pong passes followed by a quadratic loss on the final field. The passes need no value of u or v, so the adjoint keeps no snapshot of
# them; the loss reads the final u nonlinearly after every write, and the Hessian is not zero, so the second-order code is exercised too.
#
# u: field, length n, input and final state
# v: scratch field, length n, overwritten
# loss: length-1 output array, accumulated in place
# n: number of points
# nsteps: number of (u -> v, v -> u) pairs of passes
function lin_then_quad(u, v, loss, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
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
    # quadratic loss on the final field
    for i = 1:n
        loss[1] = loss[1] + 0.5 * u[i] * u[i]
    end
    return nothing
end
