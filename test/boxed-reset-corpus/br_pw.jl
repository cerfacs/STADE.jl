# br_pw(c, x, y, s1, s2, d, lam, n, maxit, tol)
#
# Power iteration with a Rayleigh quotient, every scalar boxed in a length-1 array, at most maxit steps, each run while the estimate changes.
#
# c: diagonal of the operator, length n
# x: iterate of length n
# y: work array of length n
# s1: length-1 array for x.y
# s2: length-1 array for y.y
# d: length-1 array for the change of the estimate
# lam: length-1 array for the eigenvalue estimate
# n: number of points
# maxit: iteration cap
# tol: tolerance on the change of the estimate
function br_pw(c, x, y, s1, s2, d, lam, n, maxit, tol)
    d[1] = 1.0
    lam[1] = 0.0
    for kk = 1:maxit
        if d[1] > tol
            # operator: boundary rows then the interior
            y[1] = c[1] * x[1] - x[2]
            y[n] = c[n] * x[n] - x[n - 1]
            for i = 2:n - 1
                y[i] = c[i] * x[i] - x[i - 1] - x[i + 1]
            end
            # Rayleigh quotient and the norm of y, both boxed
            s1[1] = 0.0
            for i = 1:n
                s1[1] += x[i] * y[i]
            end
            s2[1] = 0.0
            for i = 1:n
                s2[1] += y[i] * y[i]
            end
            nrm = sqrt(s2[1])
            for i = 1:n
                x[i] = y[i] / nrm * (1.0 + 0.1 * d[1])   # the previous change of the estimate scales the iterate
            end
            d[1] = abs(s1[1] - lam[1])
            lam[1] = s1[1]
        end
    end
    return nothing
end
