# br_a9(p, Ap, rs, s, out, n, maxit, j)
#
# Adversarial variant for the boxed-reset snapshot kill: the reduction target is s[j] with a variable index, so the array is not boxed.
#
# p: vector of length n
# Ap: vector of length n
# rs: length-2 array, first element is the numerator
# s: length-4 array for the dot product
# out: length-1 output
# n: length
# maxit: iterations
# j: index of the reduction target in s
function br_a9(p, Ap, rs, s, out, n, maxit, j)
    for k = 1:maxit
        s[j] = 0.0
        for i = 1:n
            s[j] += p[i] * p[i] + Ap[i] * Ap[i] + 1.0
        end
        out[1] = rs[1] / s[j]
        rs[1] = 0.3 * out[1] + 1.0
    end
    return nothing
end
