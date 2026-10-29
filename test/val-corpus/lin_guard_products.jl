# lin_guard_products(u, v, w, n, nsteps)
#
# Products and quotients whose partial derivative holds a variable. Each of them needs the value of an array that the next pass overwrites, so the
# adjoint must keep its snapshots: a product of two non-literal factors, a literal over a variable, and a literal product inside a nonlinear call.
#
# u: field, length n, overwritten each pass
# v: field, length n, overwritten each pass
# w: accumulator, length n
# n: number of points
# nsteps: number of passes
function lin_guard_products(u, v, w, n, nsteps)
    for t = 1:nsteps
        # two non-literal factors
        for i = 1:n
            v[i] = 0.5 * u[i] * u[i]
        end
        # literal numerator, variable denominator
        for i = 1:n
            u[i] = 1.0 / (1.0 + v[i])
        end
        # literal product inside a nonlinear call
        for i = 1:n
            w[i] = w[i] + 2.0 * sin(0.5 * u[i])
        end
    end
    return nothing
end
