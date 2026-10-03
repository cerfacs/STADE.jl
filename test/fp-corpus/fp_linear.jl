# fp_linear(loss, u, b, a, i_n)
#
# x = A*x + b with a diagonal contraction, so the exact fixed point is
#
#     u[i] = b[i] / (1 - 0.5*tanh(a[i]))
#
# in closed form, and so is its derivative. This is the ONLY subject whose
# gradient can be checked against algebra rather than against another
# approximation, which is why the plan makes it the gate for phases 2 and 3.
#
# tanh bounds the state coefficient below 0.5 in modulus, so the iteration
# contracts for every parameter value. Writing `0.5*a[i]*u[i]` instead would
# diverge whenever a random draw gives |a| > 2, and the validator would hang
# rather than fail.
#
# loss: length-1 output array, accumulated in place
# u: the state, length i_n
# b: source term, a parameter, length i_n
# a: contraction factor, a parameter, length i_n
# i_n: length
function fp_linear(loss, u, b, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * tanh(a[i_k]) * u[i_k] + b[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
