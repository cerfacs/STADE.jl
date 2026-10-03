# fp_nested_fp(loss, u, v, f, a, i_n)
#
# A fixed point INSIDE a fixed point, the shape of Tapenade's ala04. The
# inner state v depends on the outer state u, so the inner adjoint iteration
# must run to convergence inside EVERY outer adjoint iteration.
#
# This is the subject that forces the tape rewind to nest. One shared saved
# position would let the inner rewind destroy the outer one, and no other
# kernel in this corpus would expose that.
#
# loss: length-1 output array, accumulated in place
# u: the outer state, length i_n
# v: the inner state, length i_n
# f: source term, a parameter, length i_n
# a: coupling weight, a parameter, length i_n
# i_n: length
function fp_nested_fp(loss, u, v, f, a, i_n)
    d_out = 1.0
    while d_out > 1.0e-24
        d_out = 0.0
        d_in = 1.0
        while d_in > 1.0e-24
            d_in = 0.0
            for i_j = 1:i_n
                vn = 0.5 * v[i_j] + 0.25 * u[i_j] * tanh(a[i_j])
                d_in = d_in + (vn - v[i_j]) * (vn - v[i_j])
                v[i_j] = vn
            end
        end
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.2 * v[i_k] + 0.25 * f[i_k]
            d_out = d_out + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
