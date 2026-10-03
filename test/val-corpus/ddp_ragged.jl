# ddp_ragged(loss, u, a, v, i_n, m0)
#
# Per-sample subject whose inner segment length is reassigned inside an
# iteration-independent middle loop. That makes the snapshot stack Tier B:
# it has no closed-form trip count and resolves into per-ancestor-iteration
# prefix and value tables.
#
# The bgen_ epilogue never touches a stack. Each rank calls initstacks_*
# with its own arguments, exactly as a single-process run does. This
# kernel is the witness for that claim: if a stack ever needed a rank
# index, a ragged one would need it first.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length m0 + 2
# v: scratch activations, length i_n
# i_n: feature count
# m0: base segment length
function ddp_ragged(loss, u, a, v, i_n, m0)
    for i_x = 1:i_n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + a[i_j] * u[i_x]
            end
        end
        v[i_x] = s * s
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2]
    end
    return nothing
end
