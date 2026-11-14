# pv_a15(x, y, z, u, n)
#
# Mixed fate: one reading loop is offloadable, another is a host recurrence. Both read the same accumulated scalar.
#
# x: input of length n
# y: output of length n
# z: output of length n
# u: output of length n
# n: number of points
function pv_a15(x, y, z, u, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    u[1] = acc
    for i = 2:n
        u[i] = u[i - 1] * 0.5 + acc
    end
    return nothing
end
