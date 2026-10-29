function while_if_inside_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_kd = 0.0
    i_k = 1
    while i_k <= i_n
        if u[i_k] > 0.0
            __cse_0 = u[i_k]
            __cse_1 = a[i_k]
            vd[i_k] = __cse_0 * ad[i_k] + __cse_1 * ud[i_k]
            v[i_k] = __cse_1 * __cse_0
        else
            __cse_2 = u[i_k]
            __cse_3 = a[i_k]
            __cse_4 = (__cse_3 * __cse_2) * ud[i_k]
            vd[i_k] = ((__cse_2 * __cse_2) * ad[i_k] + __cse_4) + __cse_4
            v[i_k] = __cse_3 * __cse_2 * __cse_2
        end
        i_kd = 0.0
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        __cse_5 = v[i_x]
        lossd[1] = lossd[1] + (2__cse_5) * vd[i_x]
        loss[1] = loss[1] + __cse_5 ^ 2
    end
    return nothing
end

function while_if_inside(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        if u[i_k] > 0.0
            v[i_k] = a[i_k] * u[i_k]
        else
            v[i_k] = a[i_k] * u[i_k] * u[i_k]
        end
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
end
