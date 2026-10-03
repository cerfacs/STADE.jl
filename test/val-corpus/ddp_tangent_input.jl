# ddp_tangent_input(loss, u, a, v, i_n)
#
# Per-sample subject whose loss depends on u through a product of two
# u-dependent factors. In tangent mode the companion ud therefore enters
# the result twice, so a run that zeroes ud between samples gives a
# directional derivative for the last sample alone.
#
# ud is an input the caller loads beside u, not a buffer to clear. Only
# adjoint-valued companions take the role :local. No other corpus kernel
# reaches that distinction, because zeroing ud on a kernel with one
# u-dependent factor still leaves the adjoint correct.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# v: scratch activations, length i_n
# i_n: feature count
function ddp_tangent_input(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] * u[i_x2]
    end
    return nothing
end
