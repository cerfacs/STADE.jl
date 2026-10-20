function red_strided_d(loss, lossd, u, ud, v, vd, i_n)
    for i_x = 1:2:i_n
        __cse_0 = v[i_x]
        __cse_1 = u[i_x]
        lossd[1] = lossd[1] + (__cse_0 * ud[i_x] + __cse_1 * vd[i_x])
        loss[1] = loss[1] + __cse_1 * __cse_0
    end
    return nothing
end

function red_strided(loss, u, v, i_n)
    for i_x = 1:2:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
end
