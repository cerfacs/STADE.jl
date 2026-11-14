# sa_a1(b, x, r, p, Ap, rs, s, n, maxit, tol2)
#
# Conjugate gradient on a 1D Laplacian: the new squared residual accumulates in a local scalar inside an updating loop, under a guard in a counted loop.
#
# b: right-hand side of length n
# x: solution of length n
# r: residual of length n
# p: search direction of length n
# Ap: matrix-vector product of length n
# rs: length-1 array with r.r
# s: length-1 array with p.Ap
# n: number of points
# maxit: iteration cap
# tol2: squared residual tolerance
function sa_a1(b, x, r, p, Ap, rs, s, n, maxit, tol2)
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
            rsn = 0.0
            for i = 1:n
                x[i] += alpha * p[i]
                r[i] -= alpha * Ap[i]
                rsn += r[i] * r[i]
            end
            beta = rsn / rs[1]
            for i = 1:n
                p[i] = r[i] + beta * p[i]
            end
            rs[1] = rsn
        end
    end
    return nothing
end
