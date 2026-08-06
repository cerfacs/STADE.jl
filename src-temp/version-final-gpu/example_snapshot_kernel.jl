# sq_test(u, v, n)
#
# v[i] = u[i]^2, computed via an overwrite that forces a nonlinear
# self-reference (not a pure accumulation), so the adjoint needs a
# snapshot stack for v.
#
# u: input array
# v: output array
# n: length of u/v
function sq_test(u, v, n)
    for i_x = 1:n
        v[i_x] = u[i_x]
        v[i_x] = v[i_x] * v[i_x]
    end
    return nothing
end
