# pv_a0(x, y, z, out, n)
#
# Basic shape. A local accumulated by one loop, then read as a plain value by the next loop (the shape v0.5.18 leaves on the host).
#
# x: input of length n
# y: output of length n
# z: output of length n
# out: length-1 output
# n: number of points
function pv_a0(x, y, z, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = 3.0 * x[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    out[1] = acc
    return nothing
end
