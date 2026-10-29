# int_bound_self(x, cnt, n, out)
#
# The bound of a loop is an integer array element that the loop body rewrites. The bound is read once, when the loop is
# entered, and the reverse sweep must run the loop that many times.
#
# x: input array
# cnt: integer array of length 3; cnt[3] is the bound
# n: bound of a second loop
# out: length-1 output array; out[1] receives the sum
function int_bound_self(x, cnt, n, out)
    s = 0.0
    for i = 1:cnt[3]
        s += x[i] * x[i]
        cnt[3] = mod(cnt[3] + 2, 4) + 1
    end
    for i = 1:n
        s += x[i] * x[1]
    end
    out[1] = s
    return nothing
end
