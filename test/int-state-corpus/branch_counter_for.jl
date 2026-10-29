# branch_counter_for(x, w, n, out)
#
# A counter that advances only where x is positive, used as an index in the same pass.
#
# x: input array of length n
# w: weights of length n + 1
# n: number of entries
# out: length-1 output array; out[1] receives the sum
function branch_counter_for(x, w, n, out)
    s = 0.0
    k = 1
    for i = 1:n
        if x[i] > 0.0
            k = k + 1
        end
        s += w[k] * x[i] * x[i]
    end
    out[1] = s
    return nothing
end
