# sa_a5(a, x, y, z, out, n, m)
#
# One name, two roles: `acc` is a per-thread row sum in a loop that runs on the device, and a cross-iteration accumulator in a later loop.
#
# a: matrix of m rows and n columns, stored by rows
# x: vector of length n
# y: output of length m
# z: output of length n
# out: length-1 output
# n: columns
# m: rows
function sa_a5(a, x, y, z, out, n, m)
    for k = 1:m
        acc = 0.0
        for j = 1:n
            acc += a[(k - 1) * n + j] * x[j]
        end
        y[k] = acc
    end
    acc = 0.0
    for i = 1:n
        z[i] = x[i] * 2.0
        acc += z[i] * z[i]
    end
    out[1] = acc
    return nothing
end
