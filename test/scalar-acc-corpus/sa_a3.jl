# sa_a3(x, y, z, out, n)
#
# The accumulated scalar is also read, as a plain value, by a second loop that can run on the device: the local must not be boxed, because that kernel would receive the box.
#
# x: input of length n
# y: work array of length n
# z: output of length n
# out: length-1 output
# n: length
function sa_a3(x, y, z, out, n)
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
