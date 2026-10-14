function initstacks_red_refuse_invariant_b()
    return nothing
end

function red_refuse_invariant_hv(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd)
    for i_x = 1:i_n
        __hcse_0 = c[1]
        __hcse_1 = u[i_x]
        lossd[1] = lossd[1] + (__hcse_0 * ud[i_x] + __hcse_1 * cd[1])
        loss[1] = loss[1] + __hcse_1 * __hcse_0
    end
    for i_x = i_n:-1:1
        __cse_0d = lossbd[1]
        __cse_0 = lossb[1]
        __hcse_2 = c[1]
        ubd[i_x] = ubd[i_x] + (__cse_0 * cd[1] + __hcse_2 * __cse_0d)
        ub[i_x] = ub[i_x] + __hcse_2 * __cse_0
        __hcse_3 = u[i_x]
        cbd[1] = cbd[1] + (__cse_0 * ud[i_x] + __hcse_3 * __cse_0d)
        cb[1] = cb[1] + __hcse_3 * __cse_0
    end
    return nothing
end

function red_refuse_invariant(loss, u, c, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * c[1]
    end
end
