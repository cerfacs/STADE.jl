function initstacks_red_three_arrays_b()
    return nothing
end

function red_three_arrays_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    for i_x = 1:i_n
        __hcse_0 = w[i_x]
        __hcse_1 = v[i_x]
        __hcse_2 = u[i_x]
        __hcse_3 = __hcse_2 * __hcse_1
        lossd[1] = lossd[1] + (__hcse_0 * (__hcse_1 * ud[i_x] + __hcse_2 * vd[i_x]) + __hcse_3 * wd[i_x])
        loss[1] = loss[1] + __hcse_3 * __hcse_0
    end
    for i_x = i_n:-1:1
        __cse_0d = vd[i_x]
        __cse_0 = v[i_x]
        __cse_1d = lossbd[1]
        __cse_1 = lossb[1]
        __hcse_4 = w[i_x]
        __cse_2d = __cse_1 * wd[i_x] + __hcse_4 * __cse_1d
        __cse_2 = __hcse_4 * __cse_1
        ubd[i_x] = ubd[i_x] + (__cse_2 * __cse_0d + __cse_0 * __cse_2d)
        ub[i_x] = ub[i_x] + __cse_0 * __cse_2
        __cse_3d = ud[i_x]
        __cse_3 = u[i_x]
        vbd[i_x] = vbd[i_x] + (__cse_2 * __cse_3d + __cse_3 * __cse_2d)
        vb[i_x] = vb[i_x] + __cse_3 * __cse_2
        __hcse_5 = __cse_3 * __cse_0
        wbd[i_x] = wbd[i_x] + (__cse_1 * (__cse_0 * __cse_3d + __cse_3 * __cse_0d) + __hcse_5 * __cse_1d)
        wb[i_x] = wb[i_x] + __hcse_5 * __cse_1
    end
    return nothing
end

function red_three_arrays(loss, u, v, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * v[i_x]) * w[i_x]
    end
end
