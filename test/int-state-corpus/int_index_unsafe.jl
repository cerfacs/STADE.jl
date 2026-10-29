# int_index_unsafe(x, idx, cnt, n, out)
#
# An integer array is written at an index that is read from another integer array, which the loop rewrites. The reverse
# sweep cannot rebuild that index, so it must not replay the write.
#
# x: input array of length 6
# idx: integer array of length 6; written at cnt[1], read after the loop
# cnt: integer array of length 3; cnt[1] advances every pass
# n: number of passes
# out: length-1 output array; out[1] receives the sum
function int_index_unsafe(x, idx, cnt, n, out)
    s = 0.0
    for i = 1:n
        j = cnt[1]
        idx[j] = mod(i * 3 + 1, 6) + 1
        cnt[1] = mod(cnt[1] + 1, 4) + 1
    end
    for i = 1:n
        k = idx[i]
        s += x[k] * x[i]
    end
    out[1] = s
    return nothing
end
