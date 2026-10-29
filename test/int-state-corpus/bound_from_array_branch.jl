# bound_from_array_branch(x, cnt, out)
#
# A loop whose bound is read from an integer array, with a float branch inside. `keep_push_pop = false` has to size the
# branch stack from the bound, which the sizing function cannot know, so it must refuse.
#
# x: input array
# cnt: integer array of length 3; cnt[1] is the bound
# out: length-1 output array; out[1] receives the sum
function bound_from_array_branch(x, cnt, out)
    s = 0.0
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
