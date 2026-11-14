# br_n2(p, rs, s, out, acc, n, maxit, m)
#
# Two nested counted loops: the boxed reduction is reset in the inner loop and read again after it, in the body of the outer one.
#
# p: vector of length n
# rs: length-1 array, numerator
# s: length-1 array for the dot product
# out: length-1 output
# acc: length-1 output, accumulates over the outer loop
# n: length
# maxit: inner iterations
# m: outer iterations
function br_n2(p, rs, s, out, acc, n, maxit, m)
    for l = 1:m
        for k = 1:maxit
            s[1] = 0.0
            for i = 1:n
                s[1] += p[i] * p[i] + 0.1 * rs[1]
            end
            out[1] = rs[1] / s[1]
            rs[1] = 0.3 * out[1] + 1.0
        end
        # nonlinear read of the version the inner loop left
        acc[1] = acc[1] + 1.0 / s[1]
    end
    return nothing
end
