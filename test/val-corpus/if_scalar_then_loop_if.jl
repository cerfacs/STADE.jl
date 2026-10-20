# if_scalar_then_loop_if(u, x, y, rhs, i_n)
#
# As if_scalar_then_if, but the statement after the first `if` is an inner loop that holds an `if`. With keep_push_pop = true, through
# 0.4.9 the flags pushed inside that loop sat above the flag of the first `if`, and the generated adjoint stopped with an error. The
# inner `if` depends on the index and `u` accumulates, so every draw has a nonzero gradient and Hessian signal.
#
# u: field, length i_n, updated in place
# x: input, length >= 2
# y: output accumulator, length i_n, updated in place
# rhs: source term, length i_n
# i_n: field length
function if_scalar_then_loop_if(u, x, y, rhs, i_n)
    for i = 1:i_n
        left = 0.0
        if i > 1
            left = u[i - 1]
        end
        for j = 1:2
            if j > 1
                y[i] = y[i] + x[j] * x[j]
            end
        end
        u[i] = u[i] + rhs[i] * left * left
    end
    return nothing
end
