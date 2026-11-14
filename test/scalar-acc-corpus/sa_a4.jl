# sa_a4(x, y, out, n)
#
# Two local accumulators, one updated under a data-dependent branch, one in the other arm, in a loop that writes an array.
#
# x: input of length n
# y: output of length n
# out: length-2 output
# n: length
function sa_a4(x, y, out, n)
    a = 0.0
    b = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        if x[i] > 0.0
            a += y[i]
        else
            b -= y[i]
        end
    end
    out[1] = a
    out[2] = b
    return nothing
end
