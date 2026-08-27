# entry_branch(x, y, flag, i_n, out)
#
# Guard kernel for norm_first_touch's both-arms requirement. `s` is
# written in ONE arm of the branch, so on a pass where flag is not
# positive the value carried in from the previous pass survives and is
# read by the statement below -- the loop-entry value is LIVE. Treating a
# write in either arm as a guaranteed kill lets
# norm_insert_dead_entry_resets zero it, which is wrong by ~150%.
#
# x: input array of length i_n
# y: input array of length i_n
# flag: per-pass selector; the branch is taken where it is positive
# i_n: number of passes
# out: output array of length i_n, updated in place
function entry_branch(x, y, flag, i_n, out)
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = x[i_i] * y[i_i]
        end
        # reads the carried value on any pass the branch skipped
        out[i_i] = out[i_i] + s * s
    end
end
