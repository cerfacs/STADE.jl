# int_copy_rewritten(x, cnt, n, out)
#
# The same, and the integer array is rewritten after the loop that reads it.
#
# x: input array of length n
# cnt: integer array of length n; weights, then rewritten
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function int_copy_rewritten(x, cnt, n, out)
    s = 0.0
    for i = 1:n
        c = cnt[i]
        s += c * x[i] * x[i]
    end
    for i = 1:n
        cnt[i] = mod(cnt[i] + 1, 3) + 1
    end
    jt = cnt[1]
    s += x[jt]
    out[1] = s
    return nothing
end
