# pv_a9(x, w, y, z, out, n, m)
#
# Zero-trip: the accumulating loop runs `m` times (m may be 0) and also writes an array; the next loop reads the scalar, which is then still its start value. A loop that may not run has not run.
#
# x: input of length m
# w: output of length m
# y: input of length n
# z: output of length n
# out: length-1 output
# n: number of points
# m: trip count of the accumulating loop
function pv_a9(x, w, y, z, out, n, m)
    acc = 2.0
    for i = 1:m
        w[i] = 2.0 * x[i]
        acc += w[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    out[1] = acc
    return nothing
end
