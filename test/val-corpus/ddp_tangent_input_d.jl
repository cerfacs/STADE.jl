function ddp_tangent_input_d(loss, lossd, u, ud, a, ad, v, vd, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        __cse_1 = a[i_x]
        __cse_2 = (__cse_1 * __cse_0) * ud[i_x]
        vd[i_x] = ((__cse_0 * __cse_0) * ad[i_x] + __cse_2) + __cse_2
        v[i_x] = __cse_1 * __cse_0 * __cse_0
    end
    for i_x2 = 1:i_n
        __cse_3 = u[i_x2]
        __cse_4 = v[i_x2]
        lossd[1] = lossd[1] + (__cse_3 * vd[i_x2] + __cse_4 * ud[i_x2])
        loss[1] = loss[1] + __cse_4 * __cse_3
    end
    return nothing
end

function ddp_tangent_input(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] * u[i_x2]
    end
end
