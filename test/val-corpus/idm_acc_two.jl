# idm_acc_two(v, x, a, i_n)
#
# A scatter-accumulate loop with two different target expressions on one
# array. Both sites are commutative accumulations, so the result does not
# depend on the iteration order. Neighbouring iterations hit the same
# element: iteration i_t adds to v[i_t + 1] and iteration i_t + 1 adds to
# v[i_t + 1] again, through the other site.
#
# v: accumulator array of length i_n + 1, updated in place
# x: input array of length i_n
# a: scalar weight
# i_n: number of iterations
function idm_acc_two(v, x, a, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t] * a
        v[i_t + 1] = v[i_t + 1] + x[i_t] * x[i_t]
    end
    return nothing
end
