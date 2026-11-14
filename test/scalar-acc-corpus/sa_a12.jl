# sa_a12(x, y, out, n)
#
# An integer counter next to a float accumulator in a loop that writes an array. The counter has one shape (Int64) and must never be boxed as a Float64 array.
#
# x: input of length n
# y: output of length n
# out: length-2 output; out[1] receives the sum, out[2] the scaled counter
# n: number of points
function sa_a12(x, y, out, n)
    cnt = 0
    s = 0.0
    for i = 1:n
        y[i] = x[i] * x[i]
        s += x[i] * y[i]
        cnt += 1
    end
    out[1] = s
    out[2] = x[1] * cnt
    return nothing
end
