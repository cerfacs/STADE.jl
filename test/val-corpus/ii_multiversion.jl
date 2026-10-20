# ii_multiversion(x, w, y, loss, i_n, i_nc)
#
# A scalar reassigned three times per iteration, with a read between each assignment: `wv` is reloaded for each of three taps, like
# the weight scalar of a register-tiled convolution. The adjoint of x needs the version of `wv` that each tap read. fuse_ii_loops
# once fused the loop and rebuilt `wv` from its last assignment only, so every tap used the third weight and the gradient of x
# came out wrong (relative error about 1) in both stack modes. The classifier must refuse the loop now.
#
# x: input array of length i_n + 2
# w: weights, length 3 * i_nc
# y: output array of length i_n
# loss: length-1 output array, accumulated in place
# i_n, i_nc: trip counts of the outer and inner loop
function ii_multiversion(x, w, y, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        y[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    return nothing
end
