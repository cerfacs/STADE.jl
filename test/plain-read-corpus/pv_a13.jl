# pv_a13(x, y, z, u, v, n)
#
# Phases: accumulate, read, reset on the host, accumulate again from other data, read again. The value a reading loop sees depends on the launch order.
#
# x: input of length n
# y: output of length n
# z: output of length n
# u: output of length n
# v: output of length n
# n: number of points
function pv_a13(x, y, z, u, v, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    acc = 0.0
    for i = 1:n
        u[i] = 2.0 * x[i] + 1.0
        acc += u[i]
    end
    for i = 1:n
        v[i] = u[i] * acc
    end
    return nothing
end
