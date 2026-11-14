# pv_a4(x, y, z, u, n)
#
# The reading loop writes two arrays through a shared temporary, one of the writes reading the accumulated scalar: the loop may be split into two kernels, and only one of them reads the box.
#
# x: input of length n
# y: output of length n
# z: output of length n
# u: output of length n
# n: number of points
function pv_a4(x, y, z, u, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        t = x[i] + 1.0
        z[i] = t * acc
        u[i] = t * t
    end
    return nothing
end
