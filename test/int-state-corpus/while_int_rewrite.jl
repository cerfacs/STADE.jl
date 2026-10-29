# while_int_rewrite(x, idx, n, out)
#
# The same as for_int_rewrite, in a while loop.
#
# x: input array
# idx: integer index array of length 1; idx[1] holds the index to read, and is advanced each pass
# n: number of passes
# out: length-1 output array; out[1] receives the sum
function while_int_rewrite(x, idx, n, out)
    s = 0.0
    k = 1
    while k <= n
        j = idx[1]
        s += x[j] * x[j]
        idx[1] = idx[1] + 1
        k += 1
    end
    out[1] = s
    return nothing
end
