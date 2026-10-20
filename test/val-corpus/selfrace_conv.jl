# selfrace_conv(x, w, z, loss, i_n, i_nc)
#
# A 1-D valid convolution whose adjoint scatters into `xb` through ONE write site, `xb[t + ci - 1]`, under an inner serial loop over `ci`.
# The index is injective in the thread variable `t` for each fixed `ci`, which is all `cgen_expr_injective_ok` can see, but thread `t` at
# `ci + 1` lands on the element thread `t + 1` writes at `ci`. With no second write site to pair it against, the pairwise overlap
# analysis had nothing to compare, so the accumulation stayed a plain read-modify-write and lost updates on a real device.
#
# x: length i_n + i_nc, w: length i_nc, z: length i_n, loss: length 1
function selfrace_conv(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            a = a + x[t + ci - 1] * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
