# so_tile(x, w, b, y, z, loss, i_n, i_nc)
#
# Two register-tiled layers that reuse the same pair of operand scalars. `xv` and `wv` are reassigned for each tap with a read in
# between, so their adjoint stacks hold a save of the value each held on ENTRY to the tap: at the first tap of a tile that is the
# value the previous tile left behind, which a thread of a split tile loop does not have. The slot is dead because nothing reads
# `xv` or `wv` before it is assigned, so both tile loops must offload (adjoint and HVP). The tap loop inside must not be split
# on its own: its entry saves carry the previous tap's value, which the reverse sweep reads.
#
# x: input, length i_n + i_nc
# w: tap weights, length 2 * i_nc
# b: bias, length 1
# y: first layer output, length i_n + i_nc
# z: second layer output, length i_n
# loss: length-1 output array, accumulated in place
# i_n, i_nc: tile count and tap-block count
function so_tile(x, w, b, y, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
            xv = x[t + ci]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
        end
        y[t] = max(a + b[1], 0.0)
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = y[t + ci - 1]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
            xv = y[t + ci]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
        end
        z[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + z[k] * z[k]
    end
    return nothing
end
