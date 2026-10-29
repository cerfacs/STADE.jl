# while_two_phase(x, idx, n, out)
#
# two_phase_perm with while loops.
#
# x: input array
# idx: integer permutation, rewritten with the reversed order; its length is the integer part of n
# n: number of entries, compared as a float
# out: length-1 output array; out[1] receives the sum
function while_two_phase(x, idx, n, out)
    s = 0.0
    k = 1
    while k <= n
        j = idx[k]
        s += x[j] * x[j] * x[k]
        k += 1
    end
    k = 1
    while k <= n
        idx[k] = 5 - k
        k += 1
    end
    k = 1
    while k <= n
        j = idx[k]
        s += x[j] * x[k]
        k += 1
    end
    out[1] = s
    return nothing
end
