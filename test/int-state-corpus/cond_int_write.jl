# cond_int_write(x, idx, n, out)
#
# An index array read, then rewritten only where x is large, then read again.
#
# x: input array of length n
# idx: integer index array of length n; entries are set to 1 where x[i] > 0.5
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function cond_int_write(x, idx, n, out)
    s = 0.0
    for i = 1:n
        j = idx[i]
        s += x[j] * x[j]
    end
    for i = 1:n
        if x[i] > 0.5
            idx[i] = 1
        end
    end
    for i = 1:n
        j = idx[i]
        s += x[j] * x[i]
    end
    out[1] = s
    return nothing
end
