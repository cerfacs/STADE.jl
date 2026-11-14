# sa_a14(acc, x, y, out, n)
#
# Scalar argument accumulated and read inside the same loop.
#
# acc: scalar accumulator, starts from the argument
# x: input
# y: output
# out: length-1 output
# n: length
function sa_a14(acc, x, y, out, n)
    for i = 1:n
        acc += x[i]
        y[i] = acc * x[i]
    end
    out[1] = acc
    return nothing
end
