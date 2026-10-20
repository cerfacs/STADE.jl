function ddp_ragged_d(loss, lossd, u, ud, a, ad, v, vd, i_n, m0)
    for i_x = 1:i_n
        sd = 0.0
        s = 0.0
        for i_y = 1:2
            wd = 0.0
            w = m0 + i_y
            for i_j = 1:w
                __cse_0 = u[i_x]
                __cse_1 = a[i_j]
                sd = sd + (__cse_0 * ad[i_j] + __cse_1 * ud[i_x])
                s = s + __cse_1 * __cse_0
            end
        end
        __cse_2 = s * sd
        vd[i_x] = __cse_2 + __cse_2
        v[i_x] = s * s
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + vd[i_x2]
        loss[1] = loss[1] + v[i_x2]
    end
    return nothing
end

function ddp_ragged(loss, u, a, v, i_n, m0)
    for i_x = 1:i_n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + a[i_j] * u[i_x]
            end
        end
        v[i_x] = s * s
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2]
    end
end
