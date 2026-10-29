# while_wrapped_for(x, idx, n, out)
#
# Same computation as while_gather, with the while body wrapped in a one-trip counted loop.
#
# x: input array
# idx: integer index array, never written
# n: number of gathers
# out: length-1 output array; out[1] receives the sum
function while_wrapped_for(x, idx, n, out)
    s = 0.0
    k = 1
    while k <= n
        # one-trip counted loop around the gather
        for q = 1:1
            j = idx[k]
            s += x[j] * x[j]
        end
        k += 1
    end
    out[1] = s
    return nothing
end
