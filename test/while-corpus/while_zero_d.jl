function while_zero_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_kd = 0.0
    i_k = 1
    while i_k <= i_n
        __cse_0 = u[i_k]
        __cse_1 = a[i_k]
        __cse_2 = ud[i_k]
        vd[i_k] = (__cse_0 * ad[i_k] + __cse_1 * __cse_2) + __cse_2
        v[i_k] = __cse_1 * __cse_0 + __cse_0
        i_kd = 0.0
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        __cse_3 = v[i_x]
        lossd[1] = lossd[1] + (2__cse_3) * vd[i_x]
        loss[1] = loss[1] + __cse_3 ^ 2
    end
    return nothing
end

function while_zero(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k] + u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
