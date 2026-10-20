function idm_acc_two_d(v, vd, x, xd, a, ad, i_n)
    for i_t = 1:i_n
        __cse_0 = xd[i_t]
        __cse_1 = x[i_t]
        vd[i_t] = vd[i_t] + (a * __cse_0 + __cse_1 * ad)
        v[i_t] = v[i_t] + __cse_1 * a
        __cse_2 = __cse_1 * __cse_0
        vd[i_t + 1] = vd[i_t + 1] + (__cse_2 + __cse_2)
        v[i_t + 1] = v[i_t + 1] + __cse_1 * __cse_1
    end
    return nothing
end

function idm_acc_two(v, x, a, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t] * a
        v[i_t + 1] = v[i_t + 1] + x[i_t] * x[i_t]
    end
end
