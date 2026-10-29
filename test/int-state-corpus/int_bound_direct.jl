# int_bound_direct(x, cnt, n, out)
#
# The same as int_bound_array, with the integer array element used directly as the loop bound.
#
# x: input array
# cnt: length-1 integer array; the inner trip count, advanced each outer pass
# n: number of outer passes
# out: length-1 output array; out[1] receives the sum
function int_bound_direct(x, cnt, n, out)
    s = 0.0
    for p = 1:n
        for m = 1:cnt[1]
            s += x[m] * x[p]
        end
        cnt[1] = cnt[1] + 1
    end
    out[1] = s
    return nothing
end
