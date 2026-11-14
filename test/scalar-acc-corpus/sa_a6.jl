# sa_a6(x, y, out, n, m)
#
# A scalar that is private to each iteration: reset in the body, accumulated by an inner loop, stored per element. It is not an accumulator of the outer loop and is never boxed.
#
# x: input of length n * m
# y: output of length n
# out: length-1 output
# n: number of rows
# m: row length
function sa_a6(x, y, out, n, m)
    tot = 0.0
    for i = 1:n
        acc = 0.0
        for j = 1:m
            acc += x[(i - 1) * m + j] * x[(i - 1) * m + j]
        end
        y[i] = sqrt(acc + 1.0)
        tot += y[i]
    end
    out[1] = tot
    return nothing
end
