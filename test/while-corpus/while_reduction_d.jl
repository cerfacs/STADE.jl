function while_reduction_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_pd = 0.0
    i_p = 1
    while i_p <= 2
        sd = 0.0
        s = 0.0
        for i_k = 1:i_n
            __cse_0 = u[i_k]
            __cse_1 = a[i_k]
            sd = sd + (__cse_0 * ad[i_k] + __cse_1 * ud[i_k])
            s = s + __cse_1 * __cse_0
        end
        for i_x = 1:i_n
            __cse_2 = u[i_x]
            vd[i_x] = vd[i_x] + (__cse_2 * sd + s * ud[i_x])
            v[i_x] = v[i_x] + s * __cse_2
        end
        i_pd = 0.0
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        __cse_3 = v[i_x2]
        lossd[1] = lossd[1] + (2__cse_3) * vd[i_x2]
        loss[1] = loss[1] + __cse_3 ^ 2
    end
    return nothing
end

function while_reduction(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        s = 0.0
        for i_k = 1:i_n
            s = s + a[i_k] * u[i_k]
        end
        for i_x = 1:i_n
            v[i_x] = v[i_x] + s * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
