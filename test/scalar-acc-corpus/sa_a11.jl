# sa_a11(x, w, y, out, n, c)
#
# A subtracted term and a scalar argument in the term, with an array update in the same loop.
#
# x: input of length n
# w: weights of length n
# y: output of length n
# out: length-1 output
# n: length
# c: scalar factor
function sa_a11(x, w, y, out, n, c)
    acc = 1.0
    for i = 1:n
        y[i] = c * x[i] * w[i]
        acc -= y[i] * x[i]
    end
    out[1] = acc
    return nothing
end
