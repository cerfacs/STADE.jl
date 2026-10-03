# fp_in_outer_loop(loss, u, f, a, i_n, i_m)
#
# A fixed-point solve inside an ordinary counted time-stepping loop, solved
# afresh at each step. The adjoint iteration therefore runs once per outer
# iteration, and its tape must NOT accumulate across them: the rewind of
# section 4 has to return the tape to the same place every time.
#
# The outer loop is a plain `for`, which is what separates this subject from
# fp_nested_fp, whose outer loop is itself a fixed point.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# f: source term, a parameter, length i_n
# a: diagonal weight, a parameter, length i_n
# i_n: state length
# i_m: number of time steps
function fp_in_outer_loop(loss, u, f, a, i_n, i_m)
    for i_t = 1:i_m
        d = 1.0
        while d > 1.0e-24
            d = 0.0
            for i_k = 1:i_n
                un = 0.5 * u[i_k] + 0.25 * f[i_k] * a[i_k]
                d = d + (un - u[i_k]) * (un - u[i_k])
                u[i_k] = un
            end
        end
        for i_x = 1:i_n
            loss[1] = loss[1] + u[i_x] ^ 2
        end
    end
    return nothing
end
