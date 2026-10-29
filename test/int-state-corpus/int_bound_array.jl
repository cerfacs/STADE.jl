# int_bound_array(x, cnt, n, out)
#
# The inner trip count is read from an integer array that every outer pass rewrites in place.
#
# x: input array
# cnt: length-1 integer array; the inner trip count, advanced each outer pass
# n: number of outer passes
# out: length-1 output array; out[1] receives the sum
function int_bound_array(x, cnt, n, out)
    s = 0.0
    for p = 1:n
        c = cnt[1]
        for m = 1:c
            s += x[m] * x[p]
        end
        cnt[1] = cnt[1] + 1
    end
    out[1] = s
    return nothing
end
