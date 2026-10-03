# fp_nested_param(loss, u, f, a, i_n, i_m)
#
# The parameter enters through a nested loop rather than directly, so dG/dp
# is not a single statement. The final sweep of the adjoint, which
# accumulates the parameter gradients, therefore has real work to do rather
# than one multiply.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# f: source term, a parameter, length i_n * i_m
# a: diagonal weight, a parameter, length i_n
# i_n: state length
# i_m: inner width of the parameter
function fp_nested_param(loss, u, f, a, i_n, i_m)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            s = 0.0
            for i_j = 1:i_m
                s = s + f[(i_k - 1) * i_m + i_j]
            end
            un = 0.5 * u[i_k] + 0.25 * s * a[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
