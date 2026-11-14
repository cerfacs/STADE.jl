# pv_a2(x, y, z, out, thr, n)
#
# The accumulated scalar is read in an `if` condition inside the second loop.
#
# x: input of length n
# y: output of length n
# z: output of length n
# out: length-1 output
# thr: threshold
# n: number of points
function pv_a2(x, y, z, out, thr, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        if acc > thr
            z[i] = y[i] / acc
        else
            z[i] = y[i]
        end
    end
    out[1] = acc
    return nothing
end
