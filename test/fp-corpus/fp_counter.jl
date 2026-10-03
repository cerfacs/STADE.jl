# fp_counter(loss, u, f, a, i_n)
#
# A solver that counts its own iterations, as every Tapenade fixed-point
# test does: ala01 carries `i = i+1`, ala04 carries i1 and i2.
#
# This kernel must be ACCEPTED. The refusal for "a second carried value" is
# about ACTIVE values, not about carriage: an integer counter cannot reach
# the gradient. The first draft of the plan would have refused this, and
# with it every real solver.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# f: source term, a parameter, length i_n
# a: diagonal weight, a parameter, length i_n
# i_n: length
function fp_counter(loss, u, f, a, i_n)
    d = 1.0
    i_it = 0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * f[i_k] * a[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
        i_it = i_it + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
