# sa_pw(a, v, w, lam, n, maxit)
#
# Power iteration on a 1D Laplacian with local scalar accumulators for the squared norm and the Rayleigh numerator, both inside loops that write arrays.
#
# a: diagonal weights of length n
# v: iterate of length n
# w: work array of length n
# lam: length-1 output, Rayleigh quotient
# n: number of points
# maxit: number of iterations
function sa_pw(a, v, w, lam, n, maxit)
    for i = 1:n
        v[i] = 1.0 + a[i]
    end
    num = 0.0
    for k = 1:maxit
        w[1] = 2.0 * v[1] - v[2]
        w[n] = 2.0 * v[n] - v[n - 1]
        for i = 2:n - 1
            w[i] = 2.0 * v[i] - v[i - 1] - v[i + 1]
        end
        nrm = 0.0
        num = 0.0
        for i = 1:n
            num += w[i] * v[i]
            nrm += w[i] * w[i]
            v[i] = w[i]
        end
        sc = 1.0 / sqrt(nrm)
        for i = 1:n
            v[i] = v[i] * sc
        end
    end
    lam[1] = num
    return nothing
end
