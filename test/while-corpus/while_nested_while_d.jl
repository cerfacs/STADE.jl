function while_nested_while_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_pd = 0.0
    i_p = 1
    while i_p <= 2
        i_kd = 0.0
        i_k = 1
        while i_k <= i_n
            __cse_0 = u[i_k]
            __cse_1 = a[i_k]
            vd[i_k] = vd[i_k] + ((__cse_0 * i_p) * ad[i_k] + (__cse_1 * i_p) * ud[i_k])
            v[i_k] = v[i_k] + __cse_1 * __cse_0 * i_p
            i_kd = 0.0
            i_k = i_k + 1
        end
        i_pd = 0.0
        i_p = i_p + 1
    end
    for i_x = 1:i_n
        __cse_2 = v[i_x]
        lossd[1] = lossd[1] + (2__cse_2) * vd[i_x]
        loss[1] = loss[1] + __cse_2 ^ 2
    end
    return nothing
end

function while_nested_while(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        i_k = 1
        while i_k <= i_n
            v[i_k] = v[i_k] + a[i_k] * u[i_k] * i_p
            i_k = i_k + 1
        end
        i_p = i_p + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
