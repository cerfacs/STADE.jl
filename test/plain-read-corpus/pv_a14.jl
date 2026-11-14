# pv_a14(x, y, z, n)
#
# Control: the reading loop is a sequential recurrence, so it stays on the host. Boxing the scalar would turn every element read of that host loop into a device access: the scalar must stay a plain host scalar.
#
# x: input of length n
# y: output of length n
# z: output of length n
# n: number of points
function pv_a14(x, y, z, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    z[1] = acc
    for i = 2:n
        z[i] = z[i - 1] * 0.5 + acc
    end
    return nothing
end
