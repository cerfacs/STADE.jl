function initstacks_normcomp_b()
    return nothing
end

function normcomp_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    for i_x = 1:i_n
        wd[i_x] = ud[i_x] + -(vd[i_x])
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_x2 = 1:i_n
        __cse_1 = w[i_x2]
        lossd[1] = lossd[1] + (2__cse_1) * wd[i_x2]
        loss[1] = loss[1] + __cse_1 ^ 2
    end
    for i_x2 = i_n:-1:1
        __cse_2 = lossb[1]
        __cse_3 = 2 * w[i_x2]
        wbd[i_x2] = wbd[i_x2] + (__cse_2 * (2 * wd[i_x2]) + __cse_3 * lossbd[1])
        wb[i_x2] = wb[i_x2] + __cse_3 * __cse_2
    end
    for i_x = i_n:-1:1
        __cse_0d = wbd[i_x]
        __cse_0 = wb[i_x]
        ubd[i_x] = ubd[i_x] + __cse_0d
        ub[i_x] = ub[i_x] + __cse_0
        vbd[i_x] = vbd[i_x] + -__cse_0d
        vb[i_x] = vb[i_x] + -__cse_0
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    return nothing
end

function normcomp(loss, u, v, w, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + w[i_x2] ^ 2
    end
end
