# sa_a8(x, y, z, out, n)
#
# Two loops accumulate into the same local, with a host read between them.
#
# x: input of length n
# y: work array of length n
# z: work array of length n
# out: length-2 output
# n: length
function sa_a8(x, y, z, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] + 1.0
        acc += y[i] * y[i]
    end
    out[1] = acc
    for i = 1:n
        z[i] = y[i] * 0.5
        acc += z[i] * z[i]
    end
    out[2] = acc
    return nothing
end
