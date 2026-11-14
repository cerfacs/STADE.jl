# sa_a13(x, y, n, m)
#
# Baseline shape: a scalar that is private to each iteration (reset in the body, accumulated by an inner loop). It was offloaded before v0.5.18 and its generated code must not change.
#
# x: input of length n * m
# y: output of length n
# n: number of rows
# m: row length
function sa_a13(x, y, n, m)
    for i = 1:n
        acc = 0.0
        for j = 1:m
            acc += x[(i - 1) * m + j] * x[(i - 1) * m + j]
        end
        y[i] = sqrt(acc + 1.0)
    end
    return nothing
end
