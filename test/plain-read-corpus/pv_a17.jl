# pv_a17(a, v, w, lam, n, maxit)
#
# Power iteration whose norm accumulates in the same loop that writes the next vector, so the accumulator is boxed; the next loop divides by it (the reader sits inside the outer iteration loop).
#
# a: weights of length n
# v: iterate of length n
# w: work array of length n
# lam: length-1 output
# n: number of points
# maxit: iteration count
function pv_a17(a, v, w, lam, n, maxit)
    for i = 1:n
        v[i] = 1.0 + a[i]
    end
    nrm = 0.0
    for k = 1:maxit
        nrm = 0.0
        for i = 2:n - 1
            w[i] = 2.0 * v[i] - v[i - 1] - v[i + 1]
            nrm += w[i] * w[i]
        end
        for i = 2:n - 1
            v[i] = w[i] / sqrt(nrm + 1.0)
        end
    end
    lam[1] = nrm
    return nothing
end
