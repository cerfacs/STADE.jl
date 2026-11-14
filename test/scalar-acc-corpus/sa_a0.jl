# sa_a0(x, y, out, n)
#
# A loop that writes an array and accumulates a local scalar, read after the loop. The basic shape.
#
# x: input of length n
# y: output of length n
# out: length-1 output, receives the sum of squares of y
# n: length
function sa_a0(x, y, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = 2.0 * x[i] + 1.0
        acc += y[i] * y[i]
    end
    out[1] = acc
    return nothing
end
