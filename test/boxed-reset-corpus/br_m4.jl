# br_m4(p, Ap, rs, s, out, n, maxit)
#
# Variant for the offload study.
#
# p: vector of length n
# Ap: vector of length n
# rs: length-1 numerator
# s: length-1 array for the dot product
# out: length-1 output
# n: length
# maxit: iterations
function br_m4(p, Ap, rs, s, out, n, maxit)
    for k = 1:maxit
        s[1] = 0.0
        for i = 1:n
            s[1] += p[i] * Ap[i]
        end
        out[1] = rs[1] / s[1]
    end
    return nothing
end
