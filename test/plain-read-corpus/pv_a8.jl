# pv_a8(x, y, z, w, out, n, m)
#
# Name collision: `acc` is a private accumulator in one loop (reset per row) and the boxed, read-only value in another.
#
# x: input of length n * m
# y: output of length n
# z: output of length n
# w: input of length n
# out: length-1 output
# n: number of rows
# m: row length
function pv_a8(x, y, z, w, out, n, m)
    acc = 0.0
    for i = 1:n
        y[i] = w[i] * w[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = y[i] * acc
    end
    for i = 1:n
        acc = 0.0
        for j = 1:m
            acc += x[(i - 1) * m + j]
        end
        y[i] = y[i] + acc
    end
    out[1] = z[1]
    return nothing
end
