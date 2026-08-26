# bnd_carried(x, y, i_n, out)
#
# Guard kernel: a loop-carried scalar with NO nested write, so no block-
# boundary snapshot exists for it. The read follows the write within the
# body, so only the loop back edge justifies the write's snapshot -- and
# nothing re-establishes the value at the head of the reverse body. The
# back edge must therefore survive untouched.
#
# x: first input array of length i_n
# y: second input array of length i_n
# i_n: number of points
# out: output array of length i_n, updated in place
function bnd_carried(x, y, i_n, out)
    t = 1.0
    for i_i = 1:i_n
        t = x[i_i] * y[i_i]
        out[i_i] = out[i_i] + t * t
    end
    return nothing
end
