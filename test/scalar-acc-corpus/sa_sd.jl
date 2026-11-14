# sa_sd(b, x, r, Ar, n, maxit)
#
# Steepest descent on a 1D Laplacian; the step length needs two local accumulators that are formed in loops which also update arrays.
#
# b: right-hand side of length n
# x: solution of length n
# r: residual of length n
# Ar: matrix-vector product of length n
# n: number of points
# maxit: number of iterations
function sa_sd(b, x, r, Ar, n, maxit)
    for i = 1:n
        x[i] = 0.0
        r[i] = b[i]
    end
    for k = 1:maxit
        rr = 0.0
        for i = 1:n
            Ar[i] = 2.0 * r[i]
            rr += r[i] * r[i]
        end
        for i = 2:n - 1
            Ar[i] = 2.0 * r[i] - r[i - 1] - r[i + 1]
        end
        rAr = 0.0
        for i = 1:n
            rAr += r[i] * Ar[i]
        end
        al = rr / (rAr + 1.0)
        for i = 1:n
            x[i] = x[i] + al * r[i]
            r[i] = r[i] - al * Ar[i]
        end
    end
    return nothing
end
