# branch_counter_while(x, w, n, out)
#
# The same as branch_counter_for, in a while loop.
#
# x: input array of length n
# w: weights of length n + 1
# n: number of entries, compared as a float
# out: length-1 output array; out[1] receives the sum
function branch_counter_while(x, w, n, out)
    s = 0.0
    k = 1
    i = 1
    while i <= n
        if x[i] > 0.0
            k = k + 1
        end
        s += w[k] * x[i] * x[i]
        i += 1
    end
    out[1] = s
    return nothing
end
