# while_gather(x, idx, n, out)
#
# Sum of squares of x gathered through a fixed index array, inside a while loop.
#
# x: input array
# idx: integer index array, never written
# n: number of gathers
# out: length-1 output array; out[1] receives the sum
function while_gather(x, idx, n, out)
    s = 0.0
    k = 1
    # gather through an index array that the loop never modifies
    while k <= n
        j = idx[k]
        s += x[j] * x[j]
        k += 1
    end
    out[1] = s
    return nothing
end
