# br_a8(p, Ap, rs, s, out, n, maxit)
#
# Zero-trip variant for the boxed-reset snapshot kill.
#
# p: vector of length n
# Ap: vector of length n
# rs: length-2 array, first element is the numerator
# s: length-2 array for the dot product
# out: length-1 output
# n: length
# maxit: iterations
function br_a8(p, Ap, rs, s, out, n, maxit)
    s[1] = 0.0
    for i = 1:n
        s[1] += p[i] * p[i] + 1.0
    end
    out[1] = rs[1] / s[1]
    for k = 1:maxit
        s[1] = 0.0
        for i = 1:n
            s[1] += Ap[i] * Ap[i] + 1.0
        end
        rs[1] = rs[1] + 0.1 * s[1]
    end
    s[1] = 2.0 * rs[1] + 1.0
    out[1] = out[1] + s[1] * s[1]
    return nothing
end
