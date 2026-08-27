# ii_kill(x, u, i_n, i_w0, out, acc)
#
# Guard kernel for ii_kill_and_collect!'s loop rule. `v` is produced by
# the first loop and overwritten by the second -- but the second retires
# to a non-positive width and runs zero times, so the value that reaches
# the final statement is the FIRST loop's, which makes it an escape and
# the first loop unfusable.
#
# Treating the kill inside a possibly-empty loop as guaranteed removes v
# from the alive set, the escape goes unrecorded, the first loop is fused
# as :independent, and the gradient is wrong by ~40%. Reads are collected
# either way; only the kill is conditional.
#
# The `- 5` is load-bearing: with a width the baseline generator can only
# draw positive, the second loop always runs and the bug hides.
#
# x: input array of length i_n
# u: input array, only read when the second loop runs
# i_n: number of points
# i_w0: seed for the retired width
# out: length-1 output array, updated in place
# acc: output array of length i_n, updated in place
function ii_kill(x, u, i_n, i_w0, out, acc)
    v = 0.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        v = u[i_j] * u[i_j]
    end
    # reads the FIRST loop's v whenever the width above retired to zero
    out[1] = out[1] + v * v
end
