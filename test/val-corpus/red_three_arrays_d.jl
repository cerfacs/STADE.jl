function red_three_arrays_d(loss, lossd, u, ud, v, vd, w, wd, i_n)
    for i_x = 1:i_n
        __cse_0 = w[i_x]
        __cse_1 = v[i_x]
        __cse_2 = u[i_x]
        __cse_3 = __cse_2 * __cse_1
        lossd[1] = lossd[1] + (__cse_0 * (__cse_1 * ud[i_x] + __cse_2 * vd[i_x]) + __cse_3 * wd[i_x])
        loss[1] = loss[1] + __cse_3 * __cse_0
    end
    return nothing
end

function red_three_arrays(loss, u, v, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * v[i_x]) * w[i_x]
    end
end
