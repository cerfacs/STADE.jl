# br_sd(b, u, w, s, t, n, maxit)
#
# Steepest descent on a 1D Poisson problem, with two boxed reductions reset in the same loop body and a nonlinear use of both.
#
# b: right-hand side of length n
# u: solution of length n
# w: residual work array of length n
# s: length-1 array for the squared residual
# t: length-1 array for a weighted sum
# n: number of points
# maxit: iterations
function br_sd(b, u, w, s, t, n, maxit)
    for k = 1:maxit
        s[1] = 0.0
        t[1] = 0.0
        for i = 2:n - 1
            w[i] = b[i] - 2.0 * u[i] + u[i - 1] + u[i + 1]
            s[1] += w[i] * w[i]
            t[1] += w[i] * w[i] * (2.0 + 0.1 * u[i] * u[i])
        end
        omega = s[1] / (t[1] + 1.0)
        for i = 2:n - 1
            u[i] += omega * w[i]
        end
    end
    return nothing
end
