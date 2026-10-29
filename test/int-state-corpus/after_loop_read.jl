# after_loop_read(x, idx, n, out)
#
# An index set inside a while loop and read after it. The loop may run zero times.
#
# x: input array
# idx: integer index array
# n: number of passes, compared as a float
# out: length-1 output array; out[1] receives the sum
function after_loop_read(x, idx, n, out)
    j = 1
    k = 1
    while k <= n
        j = idx[k]
        x[j] = x[j] * 1.5
        k += 1
    end
    out[1] = x[j] * x[j]
    return nothing
end
