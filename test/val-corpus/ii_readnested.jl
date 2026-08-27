# ii_readnested(x, y, i_n, i_m, out, acc)
#
# Guard kernel for the same reverse-direction check one nesting level in,
# and the shape that decides which DIRECTION the ancestor path is walked.
# `v` is read by the acc statement and then overwritten by the nested
# loop; agen_emit_ii_loop builds vcat(fwd, bwd), so the acc adjoint in
# the backward half sees v's last inner-iteration value rather than the
# one it read going forward. Walking the ancestor path innermost-first
# instead of outermost-in misses this and restores the wrong gradient.
#
# x: first input array of length i_n
# y: second input array of length i_m
# i_n: number of outer passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
# acc: output array of length i_n, updated in place
function ii_readnested(x, y, i_n, i_m, out, acc)
    v = 1.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        # reads v, which the nested loop below then overwrites
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
    end
    return nothing
end
