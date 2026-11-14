# pv_a6(x, y, z, out, n)
#
# The reading loop is a pure reduction into a local, which has its own lowering (mapreduce): the box must appear in the reduction term.
#
# x: input of length n
# y: output of length n
# z: input of length n
# out: length-1 output
# n: number of points
function pv_a6(x, y, z, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    s = 0.0
    for i = 1:n
        s = s + z[i] * acc
    end
    out[1] = s
    return nothing
end
