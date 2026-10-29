# int_array_as_number(x, cnt, n, out)
#
# An integer array used as a multiplier in a float statement, then rewritten, then used again.
#
# x: input array of length n
# cnt: integer array of length n; incremented by 2 between the two sums
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function int_array_as_number(x, cnt, n, out)
    s = 0.0
    for i = 1:n
        s += x[i] * cnt[i]
    end
    for i = 1:n
        cnt[i] = cnt[i] + 2
    end
    for i = 1:n
        s += x[i] * x[i] * cnt[i]
    end
    out[1] = s
    return nothing
end
