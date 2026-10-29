# bound_from_written_array(x, cnt, out)
#
# A loop bound read from an integer array that the kernel rewrites before the loop, with a float branch inside.
# `keep_push_pop = false` sizes the branch stack before the kernel runs, from the array as it is on entry, which is
# not the bound the loop will have. It must refuse.
#
# x: input array
# cnt: integer array of length 3; cnt[1] + 3 is the bound
# out: length-1 output array; out[1] receives the sum
function bound_from_written_array(x, cnt, out)
    s = 0.0
    cnt[1] = cnt[1] + 3
    c = cnt[1]
    for i = 1:c
        if x[i] > 0.5
            s += x[i] * x[i]
        end
        s += x[i]
    end
    out[1] = s
    return nothing
end
