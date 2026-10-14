# idm_src_consume(u, y, i_n)
#
# A SOURCE loop (before differentiation) that the strict array rule
# refuses and the idiom rule accepts. Iteration i_t reads two slots of u and
# clears them: u[2 i_t - 1] and u[2 i_t]. The slots belong to one iteration
# each, so the loop is independent. The strict rule cannot tell this from a
# recurrence and refuses it. fuse_ii_loops must keep using the strict rule,
# so its classification of this loop must not change when the GPU gate relaxes.
#
# u: slot array of length 2 i_n, cleared in place
# y: output array of length i_n
# i_n: number of iterations
function idm_src_consume(u, y, i_n)
    for i_t = 1:i_n
        i_b = 2 * (i_t - 1)
        s = u[i_b + 1]
        u[i_b + 1] = 0.0
        r = u[i_b + 2]
        u[i_b + 2] = 0.0
        y[i_t] = s * r
    end
    return nothing
end
