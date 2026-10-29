# two_phase_perm(x, idx, n, out)
#
# Gathers through an index array, rewrites the array without reading it, then gathers through it again.
# The first gather must see the array as it was on entry.
#
# x: input array of length n
# idx: integer permutation of 1:n; overwritten with the reversed order
# n: length of x and idx
# out: length-1 output array; out[1] receives the sum
function two_phase_perm(x, idx, n, out)
    s = 0.0
    for i = 1:n
        j = idx[i]
        s += x[j] * x[j] * x[i]
    end
    for i = 1:n
        idx[i] = n - i + 1
    end
    for i = 1:n
        j = idx[i]
        s += x[j] * x[i]
    end
    out[1] = s
    return nothing
end
