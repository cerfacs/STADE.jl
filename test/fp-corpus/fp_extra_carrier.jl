# fp_extra_carrier(loss, u, f, a, r, i_n)
#
# NEGATIVE. The loop carries an ACTIVE second value across iterations: r[1]
# accumulates a residual norm that the loss then reads. The relation is no
# longer u = G(u, p), so the implicit adjoint would drop r's contribution
# silently.
#
# Contrast fp_counter, which carries an integer and is accepted. The test is
# activity, not carriage.
function fp_extra_carrier(loss, u, f, a, r, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * f[i_k] * a[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
        r[1] = r[1] + d
    end
    loss[1] = loss[1] + u[1] * u[1] + r[1]
    return nothing
end
