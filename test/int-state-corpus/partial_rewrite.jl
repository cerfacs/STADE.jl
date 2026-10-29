# partial_rewrite(x, idx, n, m, out)
#
# two_phase_perm where the rewrite covers only the first m entries. With m = 0 the rewrite loop never runs.
#
# x: input array of length n
# idx: integer permutation of 1:n; its first m entries are overwritten with the reversed order
# n: length of x and idx
# m: number of entries rewritten, 0 <= m <= n
# out: length-1 output array; out[1] receives the sum
function partial_rewrite(x, idx, n, m, out)
    s = 0.0
    for i = 1:n
        j = idx[i]
        s += x[j] * x[j] * x[i]
    end
    for i = 1:m
        idx[i] = n - i + 1
    end
    for i = 1:n
        j = idx[i]
        s += x[j] * x[i]
    end
    out[1] = s
    return nothing
end
