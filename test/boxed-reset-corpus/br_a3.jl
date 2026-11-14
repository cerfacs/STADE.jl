# br_a3(p, Ap, rs, s, out, n, maxit)
#
# Adversarial variant for the boxed-reset snapshot kill.
#
# p: vector of length n
# Ap: vector of length n
# rs: length-2 array, first element is the numerator
# s: length-2 array for the dot product
# out: length-1 output
# n: length
# maxit: iterations
function br_a3(p, Ap, rs, s, out, n, maxit)
    for k = 1:maxit
        s[1] = 0.0
        out[1] = out[1] + s[1] * s[1]
        for i = 1:n
            s[1] += p[i] * p[i] + Ap[i] * Ap[i] + 1.0
            out[1] = out[1] + 0.001 * s[1] * s[1]
        end
        out[1] = out[1] + rs[1] / s[1]
        rs[1] = 0.3 * out[1] + 1.0
    end
    return nothing
end
