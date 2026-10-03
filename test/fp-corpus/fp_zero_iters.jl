# fp_zero_iters(loss, u, f, a, i_n)
#
# The convergence test is met at entry, so the loop runs zero times. The
# adjoint iteration must then also run zero times and return the seed
# unchanged. A loop that may not run has not run, in a new place: the
# adjoint of a fixed point never reached.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# f: source term, a parameter, length i_n
# a: diagonal weight, a parameter, length i_n
# i_n: length
function fp_zero_iters(loss, u, f, a, i_n)
    d = 0.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * (f[i_k] / a[i_k])
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * a[i_x]) ^ 2
    end
    return nothing
end
