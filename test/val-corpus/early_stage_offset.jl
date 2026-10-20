# early_stage_offset(u, rhs, nu, num_levels)
#
# A smoothing stage with a bound `nc` that a later level loop reassigns (`nc = nl - 1`, with `nl` doubling). For keep_push_pop = false
# the stacks of the level loop start after the stage's slots, and the stage's size is a function of `nc`. Through 0.4.10 that size read
# the live `nc` in the forward sweep (the value of the level in progress) but a stale `nc` in the backward sweep, so the forward and the
# backward index of every level-loop slot differed: a wrong gradient without any error (relative error about 1), or a BoundsError when
# `nc` had shrunk. The sizing pass of initstacks_* now captures the stage's size where the stage stands, as `__mult_<k>`.
#
# u: grid, size (2^num_levels, num_levels), updated in place
# rhs: right-hand side, size (1, 1) or larger
# nu: number of smoothing sweeps
# num_levels: number of levels, at least 2
function early_stage_offset(u, rhs, nu, num_levels)
    nl = 2
    nc = nl - 1
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            u[i_j, 1] = 0.5 * (rhs[i_j, 1] + left) * left
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        nc = nl - 1
        for j = 1:nl
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            u[j, i_level] = u[j, i_level - 1] * cl
        end
    end
    return nothing
end
