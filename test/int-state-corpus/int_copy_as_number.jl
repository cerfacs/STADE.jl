# int_copy_as_number(x, cnt, n, out)
#
# An integer array element copied to a scalar that is only used as a number in a float statement. The scalar is an
# integer, and the reverse sweep must recompute it for every iteration (it used to be typed as a float and held the
# value of the last iteration).
#
# x: input array of length n
# cnt: integer array of length n; weights
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function int_copy_as_number(x, cnt, n, out)
    s = 0.0
    for i = 1:n
        c = cnt[i]
        s += c * x[i] * x[i]
    end
    jt = cnt[1]
    s += x[jt]
    out[1] = s
    return nothing
end
