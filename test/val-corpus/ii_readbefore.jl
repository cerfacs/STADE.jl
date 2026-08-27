# ii_readbefore(x, i_n, i_m, out)
#
# Guard kernel for fuse_ii_loops' reverse-direction escape check
# (ii_read_reaches_from_before). The statement before the loop reads `s`,
# and the loop then overwrites it. That read's adjoint runs AFTER the
# loop's backward code, and every fusing kind elides its fused scalars'
# snapshots -- so classifying this loop leaves nothing to re-establish
# the value the read needs. Fusing it produced a wrong gradient of
# roughly 150% relative error, visible to both the finite-difference and
# the exact dot-product oracles.
#
# x: input array of length max(i_n, i_m)
# i_n: number of outer passes
# i_m: inner segment length
# out: output array of length i_n, updated in place
function ii_readbefore(x, i_n, i_m, out)
    s = x[1] * x[1]
    # reads s before the loop below overwrites it
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
    return nothing
end
