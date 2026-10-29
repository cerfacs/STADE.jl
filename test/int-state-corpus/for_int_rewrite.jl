# for_int_rewrite(x, idx, n, out)
#
# Sum of squares of x gathered through an index array that every pass of a for loop rewrites in place.
# The element read by one pass is not the element read by the next.
#
# x: input array
# idx: integer index array of length 1; idx[1] holds the index to read, and is advanced each pass
# n: number of passes
# out: length-1 output array; out[1] receives the sum
function for_int_rewrite(x, idx, n, out)
    s = 0.0
    for k = 1:n
        j = idx[1]
        s += x[j] * x[j]
        idx[1] = idx[1] + 1
    end
    out[1] = s
    return nothing
end
