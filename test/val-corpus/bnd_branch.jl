# bnd_branch(x, flag, i_n, i_m, out)
#
# Guard kernel: the block-boundary snapshot on an `if`-branch body rather
# than a loop body. agen_forward_body emits a boundary push at the end of
# every body it walks, branches included, so the kill has to handle a
# branch body the same way -- and a pass in which the branch is not taken
# must leave the scalar alone.
#
# x: input array of length max(i_n, i_m)
# flag: per-pass selector; the branch is taken where it is positive
# i_n: number of passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
function bnd_branch(x, flag, i_n, i_m, out)
    s = 0.0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = 0.0
            for i_j = 1:i_m
                s = s + x[i_j] * x[i_i]
            end
            out[i_i] = out[i_i] + s * s
        end
    end
    return nothing
end
