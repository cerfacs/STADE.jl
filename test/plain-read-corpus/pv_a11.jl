# pv_a11(b, x, r, p, Ap, rs, n, maxit)
#
# Conjugate gradient with the direction update written directly from the accumulated scalars (no host scalar `beta`): the accumulator is read by the next loop in every outer iteration.
#
# b: right-hand side of length n
# x: solution of length n
# r: residual of length n
# p: search direction of length n
# Ap: matrix-vector product of length n
# rs: length-1 array with r.r
# n: number of points
# maxit: iteration count
function pv_a11(b, x, r, p, Ap, rs, n, maxit)
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
        for i = 1:n
            x[i] += rs[1] / (pAp + 1.0) * p[i]
            r[i] -= rs[1] / (pAp + 1.0) * Ap[i]
        end
        rsn = 0.0
        for i = 1:n
            rsn += r[i] * r[i]
        end
        for i = 1:n
            p[i] = r[i] + rsn / (rs[1] + 1.0) * p[i]
        end
        rs[1] = rsn
    end
    return nothing
end
