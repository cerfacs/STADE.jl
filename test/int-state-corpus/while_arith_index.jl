# while_arith_index(x, y, n0, nmax, out)
#
# Geometric growth y[n+1] = x[1] * y[n], extended one entry per while iteration.
# The destination index is held in a loop-local integer temporary.
#
# x: input array; x[1] is the growth factor
# y: work array of length nmax; y[n0] holds the starting value on entry
# n0: index of the starting value
# nmax: index of the last entry to compute
# out: length-1 output array; out[1] receives y[nmax]
function while_arith_index(x, y, n0, nmax, out)
    n = n0
    # n is carried between iterations
    while n < nmax
        j = n + 1
        y[j] = x[1] * y[n]
        n += 1
    end
    out[1] = y[nmax]
    return nothing
end
