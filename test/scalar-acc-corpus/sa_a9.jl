# sa_a9(x, y, u, out, n)
#
# The accumulated local is read inside a sequential loop (a recurrence) that stays on the host.
#
# x: input of length n
# y: work array of length n
# u: output of length n, filled by a recurrence
# out: length-1 output
# n: length
function sa_a9(x, y, u, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    u[1] = x[1]
    # recurrence: each point reads the previous one
    for i = 2:n
        u[i] = u[i - 1] * 0.5 + acc * x[i]
    end
    out[1] = acc
    return nothing
end
