# nest_if_scalar(u, rhs, i_nu, i_n)
#
# An in-place stencil sweep whose body is wrapped in two `if`s, with two inner `if`s that give the scalars `left` and `right` their values.
# Through 0.4.9 this failed in two ways. With keep_push_pop = true the flag order was wrong and the generated adjoint stopped with an
# error. With either setting of keep_push_pop the backward sweep recomputed `left` and `right` from `u` after it had restored them from
# their stacks, and `u` no longer held the state the forward read had seen (relative error about 0.2). The update holds the product
# `left * right`, so the Hessian check has a signal.
#
# u: field, length i_n, updated in place
# rhs: source term, length i_n
# i_nu: sweep count
# i_n: field length
function nest_if_scalar(u, rhs, i_nu, i_n)
    for i_k = 1:i_nu
        if i_nu >= 0
            for i_j = 1:i_n
                if i_nu >= 0
                    left = 0.0
                    if i_j > 1
                        left = u[i_j - 1]
                    end
                    right = 0.0
                    if i_j < i_n
                        right = u[i_j + 1]
                    end
                    u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
                end
            end
        end
    end
    return nothing
end
