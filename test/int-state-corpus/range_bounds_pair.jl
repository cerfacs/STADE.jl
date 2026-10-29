# range_bounds_pair(x, rp, n, out)
#
# Ragged sums: row v owns the entries rp[v] .. rp[v+1] - 1 of x. Both ends of the inner range are
# scalars assigned in the outer loop body, so the reverse sweep keeps both on the trip-count stack.
# It must pop them in the reverse of the order it pushed them, or the reverse loop runs over the
# wrong range (it used to, whenever a loop had two reassigned bounds).
#
# x: input array
# rp: integer array of length n + 1, row pointers, rp[1] = 1
# n: number of rows
# out: length-1 output array; out[1] receives the sum
function range_bounds_pair(x, rp, n, out)
    s = 0.0
    for v = 1:n
        lo = rp[v]
        hi = rp[v + 1] - 1
        t = 0.0
        for e = lo:hi
            t += x[e] * x[e]
        end
        s += t * t
    end
    out[1] = s
    return nothing
end
