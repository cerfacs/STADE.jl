function while_active_cond_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    sd = 0.0
    s = 1.0
    i_kd = 0.0
    i_k = 1
    while s > 0.05 && i_k <= i_n
        __cse_0 = a[i_k]
        __cse_1 = 0.5 + 0.25 * __cse_0 * __cse_0
        __cse_2 = (0.25__cse_0) * ad[i_k]
        sd = __cse_1 * sd + s * (__cse_2 + __cse_2)
        s = s * __cse_1
        __cse_3 = u[i_k]
        vd[i_k] = __cse_3 * sd + s * ud[i_k]
        v[i_k] = s * __cse_3
        i_kd = 0.0
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        __cse_4 = v[i_x]
        lossd[1] = lossd[1] + (2__cse_4) * vd[i_x]
        loss[1] = loss[1] + __cse_4 ^ 2
    end
    return nothing
end

function while_active_cond(loss, u, a, v, i_n)
    s = 1.0
    i_k = 1
    while s > 0.05 && i_k <= i_n
        s = s * (0.5 + 0.25 * a[i_k] * a[i_k])
        v[i_k] = s * u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
