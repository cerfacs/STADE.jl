# sa_a7(x, c, y, out, n, m)
#
# A zero-trip shape: the inner loop accumulates into a local that the outer loop reads every pass, and either bound can be zero.
#
# x: input of length n
# c: factors of length m
# y: work array of length n
# out: output of length m
# n: inner length
# m: outer length
function sa_a7(x, c, y, out, n, m)
    acc = 0.0
    for k = 1:m
        for i = 1:n
            y[i] = x[i] * c[k]
            acc += y[i] * y[i]
        end
        out[k] = acc
    end
    return nothing
end
