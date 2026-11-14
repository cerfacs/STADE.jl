# sa_a2(x, y, out, n)
#
# The accumulator is read inside its own loop (a running sum): the iterations are order dependent, so the loop must stay sequential.
#
# x: input of length n
# y: output of length n
# out: length-1 output
# n: length
function sa_a2(x, y, out, n)
    acc = 0.0
    for i = 1:n
        acc += x[i]
        y[i] = acc * x[i]
    end
    out[1] = acc
    return nothing
end
