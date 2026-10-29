function while_for_inside_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    i_pd = 0.0
    i_p = 1
    while i_p <= 2
        for i_x = 1:i_n
            __cse_0 = u[i_x]
            __cse_1 = a[i_x]
            vd[i_x] = vd[i_x] + (__cse_0 * ad[i_x] + __cse_1 * ud[i_x])
            v[i_x] = v[i_x] + __cse_1 * __cse_0
        end
        i_pd = 0.0
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        __cse_2 = v[i_x2]
        lossd[1] = lossd[1] + (2__cse_2) * vd[i_x2]
        loss[1] = loss[1] + __cse_2 ^ 2
    end
    return nothing
end

function while_for_inside(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        for i_x = 1:i_n
            v[i_x] = v[i_x] + a[i_x] * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
