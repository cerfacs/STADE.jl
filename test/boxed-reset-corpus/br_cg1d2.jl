# br_cg1d2(b, x, r, p, Ap, rs, s, t, n, maxit, tol2)
#
# Conjugate gradient on the 1D Laplacian, at most maxit iterations.
#
# b: right-hand side of length n
# x: solution of length n, starts at zero
# r: residual work array of length n
# p: search direction work array of length n
# Ap: matrix-vector product work array of length n
# rs: length-1 array holding r.r
# s: length-1 array holding p.Ap
# n: number of points
# maxit: iteration cap
# t: length-1 array holding the new r.r
# tol2: squared residual tolerance
function br_cg1d2(b, x, r, p, Ap, rs, s, t, n, maxit, tol2)
    rs[1] = 0.0
    for i = 1:n
        r[i] = b[i]
        p[i] = b[i]
        x[i] = 0.0
        Ap[i] = 0.0
        rs[1] += b[i] * b[i]
    end
    for k = 1:maxit
        if rs[1] > tol2
            for i = 2:n - 1
                Ap[i] = 2.0 * p[i] - p[i - 1] - p[i + 1]
            end
            s[1] = 0.0
            for i = 1:n
                s[1] += p[i] * Ap[i]
            end
            alpha = rs[1] / s[1]
            t[1] = 0.0
            for i = 1:n
                x[i] += alpha * p[i]
                r[i] -= alpha * Ap[i]
                t[1] += r[i] * r[i]
            end
            beta = t[1] / rs[1]
            for i = 1:n
                p[i] = r[i] + beta * p[i]
            end
            rs[1] = t[1]
        end
    end
    return nothing
end
