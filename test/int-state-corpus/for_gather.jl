# for_gather(x, idx, n, out)
#
# Same computation as while_gather, in a counted loop. Control case.
#
# x: input array
# idx: integer index array, never written
# n: number of gathers
# out: length-1 output array; out[1] receives the sum
function for_gather(x, idx, n, out)
    s = 0.0
    for k = 1:n
        j = idx[k]
        s += x[j] * x[j]
    end
    out[1] = s
    return nothing
end
