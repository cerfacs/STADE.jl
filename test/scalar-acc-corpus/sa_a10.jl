# sa_a10(x, y, out, n)
#
# Not an accumulation: the scalar is rescaled before the add, so each iteration depends on the previous one and the loop must stay sequential.
#
# x: input of length n
# y: output of length n
# out: length-1 output
# n: length
function sa_a10(x, y, out, n)
    acc = 0.0
    for i = 1:n
        y[i] = x[i] * 2.0
        acc = acc * 0.5 + y[i]
    end
    out[1] = acc
    return nothing
end
