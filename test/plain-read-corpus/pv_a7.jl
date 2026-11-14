# pv_a7(x, y, z, out, n)
#
# The reading loop accumulates into an ARRAY element (the large-trip branch has its own kernel and its own copy of the term).
#
# x: input of length n
# y: output of length n
# z: input of length n
# out: length-1 output
# n: number of points
function pv_a7(x, y, z, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    out[1] = 0.0
    for i = 1:n
        out[1] = out[1] + z[i] * acc
    end
    return nothing
end
