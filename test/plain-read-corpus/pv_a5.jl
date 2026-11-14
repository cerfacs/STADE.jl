# pv_a5(x, y, z, out, n)
#
# The reading loop also accumulates a second local: one box is read, the other is added to.
#
# x: input of length n
# y: output of length n
# z: output of length n
# out: length-2 output
# n: number of points
function pv_a5(x, y, z, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    nrm = 0.0
    for i = 1:n
        z[i] = y[i] * acc
        nrm += z[i] * z[i]
    end
    out[1] = acc
    out[2] = nrm
    return nothing
end
