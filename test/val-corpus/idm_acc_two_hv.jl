function initstacks_idm_acc_two_b()
    return nothing
end

function idm_acc_two_hv(v, vb, x, xb, a, ab, i_n, vd, vbd, xd, xbd, ad, abd)
    for i_t = 1:i_n
        __cse_0d = xd[i_t]
        __cse_0 = x[i_t]
        vd[i_t] = vd[i_t] + (a * __cse_0d + __cse_0 * ad)
        v[i_t] = v[i_t] + __cse_0 * a
        __hcse_0 = __cse_0 * __cse_0d
        vd[i_t + 1] = vd[i_t + 1] + (__hcse_0 + __hcse_0)
        v[i_t + 1] = v[i_t + 1] + __cse_0 * __cse_0
    end
    for i_t = i_n:-1:1
        __cse_1d = xd[i_t]
        __cse_1 = x[i_t]
        __hcse_1 = vb[i_t + 1]
        __cse_2d = __hcse_1 * __cse_1d + __cse_1 * vbd[i_t + 1]
        __cse_2 = __cse_1 * __hcse_1
        xbd[i_t] = xbd[i_t] + __cse_2d
        xb[i_t] = xb[i_t] + __cse_2
        xbd[i_t] = xbd[i_t] + __cse_2d
        xb[i_t] = xb[i_t] + __cse_2
        __cse_3d = vbd[i_t]
        __cse_3 = vb[i_t]
        xbd[i_t] = xbd[i_t] + (__cse_3 * ad + a * __cse_3d)
        xb[i_t] = xb[i_t] + a * __cse_3
        abd = abd + (__cse_3 * __cse_1d + __cse_1 * __cse_3d)
        ab = ab + __cse_1 * __cse_3
    end
    return (ab, abd)
end

function idm_acc_two(v, x, a, i_n)
    for i_t = 1:i_n
        v[i_t] = v[i_t] + x[i_t] * a
        v[i_t + 1] = v[i_t + 1] + x[i_t] * x[i_t]
    end
end
