# idm_refuse_cond_read(v, z, x, i_n)
#
# Must stay on the host. Iteration i_t accumulates into v[i_t] and
# v[i_t + 1], then branches on v[i_t]. Iteration i_t - 1 also added to
# v[i_t], so the branch sees a value that depends on the iteration order.
# The accumulation is commutative, but the read in the condition is not.
#
# v: accumulator array of length i_n + 1, updated in place
# z: output array of length i_n
# x: input array of length i_n
# i_n: number of iterations
function idm_refuse_cond_read(v, z, x, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t]
        v[i_t + 1] = v[i_t + 1] + x[i_t] * 2.0
        # the branch reads an element that a neighbouring iteration wrote
        if v[i_t] > 0.0
            z[i_t] = x[i_t] * 3.0
        else
            z[i_t] = x[i_t]
        end
    end
    return nothing
end
