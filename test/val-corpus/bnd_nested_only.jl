# bnd_nested_only(x, y, i_n, i_m, out, acc)
#
# Guard kernel: a scalar written ONLY inside a nested loop, never at the
# top level of the body that snapshots it, and written again after the
# loop. The kill must key on the body that actually carries the boundary
# snapshot; a kill applied to every candidate scalar at every body end
# retires the post-loop write's push, which the reverse sweep still needs
# to re-establish the loop's final value.
#
# x: first input array of length i_m
# y: second input array of length i_n
# i_n: number of outer passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
# acc: length-1 output array holding the trailing term
function bnd_nested_only(x, y, i_n, i_m, out, acc)
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            v = x[i_j] * y[i_i]
            out[i_i] = out[i_i] + v * v
        end
        # sequential coupling: keeps this loop off the fusion path
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
    end
    v = x[1] * y[1]
    acc[1] = acc[1] + v * v
    return nothing
end
