# pv_a12(a, v, w, lam, n, maxit)
#
# Power iteration: the vector is normalized directly by the accumulated squared norm of the previous loop (the most common use of the shape).
#
# a: diagonal weights of length n
# v: iterate of length n
# w: work array of length n
# lam: length-1 output
# n: number of points
# maxit: iteration count
function pv_a12(a, v, w, lam, n, maxit)
    for i = 1:n
        v[i] = 1.0 + a[i]
    end
    nrm = 0.0
    for k = 1:maxit
        w[1] = 2.0 * v[1] - v[2]
        w[n] = 2.0 * v[n] - v[n - 1]
        for i = 2:n - 1
            w[i] = 2.0 * v[i] - v[i - 1] - v[i + 1]
        end
        nrm = 0.0
        for i = 1:n
            nrm += w[i] * w[i]
        end
        for i = 1:n
            v[i] = w[i] / sqrt(nrm)
        end
    end
    lam[1] = nrm
    return nothing
end
