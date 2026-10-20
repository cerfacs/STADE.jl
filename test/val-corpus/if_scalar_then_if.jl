# if_scalar_then_if(u, x, y, rhs, i_n)
#
# An `if` that gives the scalar `left` its value, followed by a second `if` in the same loop body. With keep_push_pop = true, through
# 0.4.9 the backward sweep popped the flag of the first `if` ahead of the rest of its block, so it took the flag of the second one
# (wrong gradient, relative error about 1). The second `if` depends on the index and `u` accumulates, so every draw has a nonzero
# gradient and Hessian signal.
#
# u: field, length i_n, updated in place
# x: input, length i_n
# y: output accumulator, length i_n, updated in place
# rhs: source term, length i_n
# i_n: field length
function if_scalar_then_if(u, x, y, rhs, i_n)
    for i = 1:i_n
        left = 0.0
        if i > 1
            left = u[i - 1]
        end
        if i > 2
            y[i] = y[i] + x[i] * x[i]
        end
        u[i] = u[i] + rhs[i] * left * left
    end
    return nothing
end
