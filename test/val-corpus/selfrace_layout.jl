# selfrace_layout(x, w, z, loss, i_hw, i_c)
#
# Negative control for selfrace_conv. The adjoint scatters into `xb` through one site again, `xb[(ci - 1) * i_hw + t]` under an inner loop over
# `ci`, but here the stride is the thread range: each channel is a separate slice, so no two threads ever meet. The offset moves with `ci`
# through a SYMBOLIC stride, which says nothing about overlap either way, and the plain write must stay plain (an atomic would be correct
# and slower on the layouts every conv uses).
#
# x, z: length i_hw * i_c, w: length i_c, loss: length 1
function selfrace_layout(x, w, z, loss, i_hw, i_c)
    for t = 1:i_hw
        for ci = 1:i_c
            z[(ci - 1) * i_hw + t] = z[(ci - 1) * i_hw + t] + x[(ci - 1) * i_hw + t] * w[ci]
        end
    end
    for m = 1:i_hw * i_c
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
