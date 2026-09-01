function initstacks_weightedsumsq_b()
    return nothing
end

function weightedsumsq_hv(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    for i_x = 1:i_n
        __cse_2 = u[i_x]
        __cse_3 = __cse_2 ^ 2
        __cse_4 = w[i_x]
        lossd[1] = lossd[1] + (__cse_3 * wd[i_x] + __cse_4 * ((2__cse_2) * ud[i_x]))
        loss[1] = loss[1] + __cse_4 * __cse_3
    end
    for i_x = i_n:-1:1
        __cse_0d = ud[i_x]
        __cse_0 = u[i_x]
        __cse_1d = lossbd[1]
        __cse_1 = lossb[1]
        __cse_5 = 2__cse_0
        __cse_6 = __cse_0 ^ 2
        wbd[i_x] = wbd[i_x] + (__cse_1 * (__cse_5 * __cse_0d) + __cse_6 * __cse_1d)
        wb[i_x] = wb[i_x] + __cse_6 * __cse_1
        __cse_7 = w[i_x]
        __cse_8 = __cse_7 * __cse_1
        ubd[i_x] = ubd[i_x] + (__cse_8 * (2__cse_0d) + __cse_5 * (__cse_1 * wd[i_x] + __cse_7 * __cse_1d))
        ub[i_x] = ub[i_x] + __cse_5 * __cse_8
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
end
