# while_assign_cond(loss, u, v, i_n)
#
# NEGATIVE. The condition contains an assignment. parse_ must refuse it:
# a condition is an expression, and a side effect inside one has no place
# in the forward sweep, where the condition is evaluated once per
# iteration, or in the backward sweep, where it is never evaluated at all.
#
# This file is not valid STADE input and is never differentiated. It exists
# so the refusal has a witness.
function while_assign_cond(loss, u, v, i_n)
    i_k = 1
    while (i_k = i_k + 1) <= i_n
        v[i_k] = u[i_k]
    end
    loss[1] = loss[1] + v[1]
    return nothing
end
