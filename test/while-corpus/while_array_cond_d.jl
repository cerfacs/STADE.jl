function while_array_cond_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_kd = 0.0
    i_k = 1
    vd[1] = 0.0
    v[1] = 1.0
    while v[i_k] > 0.1 && i_k < i_n
        __cse_0 = a[i_k]
        __cse_1 = 1.0 + __cse_0 * __cse_0
        __cse_2 = v[i_k]
        __cse_3 = __cse_0 * ad[i_k]
        vd[i_k + 1] = (0.5__cse_1) * vd[i_k] + (0.5__cse_2) * (__cse_3 + __cse_3)
        v[i_k + 1] = 0.5 * __cse_2 * __cse_1
        i_kd = 0.0
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        __cse_4 = v[i_x]
        __cse_5 = u[i_x]
        __cse_6 = __cse_4 * __cse_5
        lossd[1] = lossd[1] + (2__cse_6) * (__cse_5 * vd[i_x] + __cse_4 * ud[i_x])
        loss[1] = loss[1] + __cse_6 ^ 2
    end
    return nothing
end

function while_array_cond(loss, u, a, v, i_n)
    i_k = 1
    v[1] = 1.0
    while v[i_k] > 0.1 && i_k < i_n
        v[i_k + 1] = 0.5 * v[i_k] * (1.0 + a[i_k] * a[i_k])
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + (v[i_x] * u[i_x]) ^ 2
    end
end
