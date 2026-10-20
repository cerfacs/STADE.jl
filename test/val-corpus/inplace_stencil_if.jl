# inplace_stencil_if(u, rhs, i_nu, i_n)
#
# An in-place sweep: `u[i_j]` is rebuilt from `u[i_j - 1]` (already updated) and `u[i_j + 1]` (not yet), each read through a scalar that
# an `if` assigns at the array's ends. The adjoint needs the values of `left` and `right`, which it restores from stacks. Through 0.4.9
# the backward sweep then recomputed both scalars from `u`, which by then held the final sweep state, overwriting the restored values (a
# dot-product mismatch of 1e-3 to 1e-1 depending on the draw, in both keep_push_pop settings). The tangent and HVP checks did not see it.
#
# u: field, length i_n, updated in place
# rhs: source term, length i_n
# i_nu: sweep count
# i_n: field length
function inplace_stencil_if(u, rhs, i_nu, i_n)
    for i_k = 1:i_nu
        for i_j = 1:i_n
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
    return nothing
end
