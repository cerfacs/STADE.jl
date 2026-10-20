function initstacks_idm_acc_two_b()
    return nothing
end

function idm_acc_two_b(v, vb, x, xb, a, ab, i_n)
    for i_t = 1:i_n
        __cse_0 = x[i_t]
        v[i_t] = v[i_t] + __cse_0 * a
        v[i_t + 1] = v[i_t + 1] + __cse_0 * __cse_0
    end
    for i_t = i_n:-1:1
        __cse_1 = x[i_t]
        __cse_2 = __cse_1 * vb[i_t + 1]
        xb[i_t] = xb[i_t] + __cse_2
        xb[i_t] = xb[i_t] + __cse_2
        __cse_3 = vb[i_t]
        xb[i_t] = xb[i_t] + a * __cse_3
        ab = ab + __cse_1 * __cse_3
    end
    return ab
end

function idm_acc_two(v, x, a, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t] * a
        v[i_t + 1] = v[i_t + 1] + x[i_t] * x[i_t]
    end
end
