# pv_a16(x, y, out, n)
#
# Control: the loop reads its own accumulator (a running sum). It is order-dependent and stays on the host with or without this change (reason :accread).
#
# x: input of length n
# y: output of length n
# out: length-1 output
# n: number of points
function pv_a16(x, y, out, n)
    acc = 0.0
    for i = 1:n
        acc += x[i]
        y[i] = acc * x[i]
    end
    out[1] = acc
    return nothing
end
