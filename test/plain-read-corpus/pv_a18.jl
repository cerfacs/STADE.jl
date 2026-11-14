# pv_a18(b, x, r, p, Ap, rs, n, maxit)
#
# Conjugate gradient whose squared residual accumulates beside the residual update (boxed), and whose direction update reads it directly in the next loop. Both scalars are read in later loops of the outer iteration.
#
# b: right-hand side of length n
# x: solution of length n
# r: residual of length n
# p: search direction of length n
# Ap: matrix-vector product of length n
# rs: length-1 array with r.r
# n: number of points
# maxit: iteration count
function pv_a18(b, x, r, p, Ap, rs, n, maxit)
    rs[1] = 0.0
    for i = 1:n
        r[i] = b[i]
        p[i] = b[i]
        x[i] = 0.0
        Ap[i] = 0.0
        rs[1] += b[i] * b[i]
    end
    for k = 1:maxit
        for i = 2:n - 1
            Ap[i] = 2.0 * p[i] - p[i - 1] - p[i + 1]
        end
        pAp = 0.0
        for i = 1:n
            pAp += p[i] * Ap[i]
        end
        alpha = rs[1] / (pAp + 1.0)
        rsn = 0.0
        for i = 1:n
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            rsn += r[i] * r[i]
        end
        for i = 1:n
            p[i] = r[i] + rsn / (rs[1] + 1.0) * p[i]
        end
        rs[1] = rsn
    end
    return nothing
end
