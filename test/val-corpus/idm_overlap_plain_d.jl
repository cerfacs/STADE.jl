function idm_overlap_plain_d(a, ad, x, xd, y, yd, loss, lossd, i_n)
    for i_t = 1:i_n
        ad[i_t] = 2.0 * xd[i_t]
        a[i_t] = x[i_t] * 2.0
        ad[i_t + 1] = 3.0 * yd[i_t]
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        __cse_0 = a[i_k]
        __cse_1 = __cse_0 * ad[i_k]
        lossd[1] = lossd[1] + (__cse_1 + __cse_1)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    return nothing
end

function idm_overlap_plain(a, x, y, loss, i_n)
    for i_t = 1:i_n
        a[i_t] = x[i_t] * 2.0
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        loss[1] = loss[1] + a[i_k] * a[i_k]
    end
end
