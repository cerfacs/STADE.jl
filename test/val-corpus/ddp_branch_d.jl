function ddp_branch_d(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __cse_0 = u[i_x]
            __cse_1 = a[i_x]
            vd[i_x] = __cse_0 * ad[i_x] + __cse_1 * ud[i_x]
            v[i_x] = __cse_1 * __cse_0
        else
            __cse_2 = u[i_x]
            __cse_3 = b[i_x]
            __cse_4 = (__cse_3 * __cse_2) * ud[i_x]
            vd[i_x] = ((__cse_2 * __cse_2) * bd[i_x] + __cse_4) + __cse_4
            v[i_x] = __cse_3 * __cse_2 * __cse_2
        end
    end
    for i_x2 = 1:i_n
        __cse_5 = v[i_x2]
        lossd[1] = lossd[1] + (2__cse_5) * vd[i_x2]
        loss[1] = loss[1] + __cse_5 ^ 2
    end
    return nothing
end

function ddp_branch(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = a[i_x] * u[i_x]
        else
            v[i_x] = b[i_x] * u[i_x] * u[i_x]
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
