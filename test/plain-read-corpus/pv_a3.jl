# pv_a3(x, y, w, n, m)
#
# The accumulated scalar is read inside a device inner loop of the second loop.
#
# x: input of length n
# y: output of length n
# w: output of length n * m
# n: number of rows
# m: columns per row
function pv_a3(x, y, w, n, m)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        acc += y[i]
    end
    for i = 1:n
        for j = 1:m
            w[(i - 1) * m + j] = y[i] * acc * j
        end
    end
    return nothing
end
