function initstacks_stencil_loss_b()
    return nothing
end

function stencil_loss_hv(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    for i_x = 2:i_n - 1
        wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_x2 = 2:i_n - 1
        __cse_1 = w[i_x2]
        lossd[1] = lossd[1] + (2__cse_1) * wd[i_x2]
        loss[1] = loss[1] + __cse_1 ^ 2
    end
    for i_x2 = i_n - 1:-1:2
        __cse_2 = lossb[1]
        __cse_3 = 2 * w[i_x2]
        wbd[i_x2] = wbd[i_x2] + (__cse_2 * (2 * wd[i_x2]) + __cse_3 * lossbd[1])
        wb[i_x2] = wb[i_x2] + __cse_3 * __cse_2
    end
    for i_x = i_n - 1:-1:2
        __cse_0d = wbd[i_x]
        __cse_0 = wb[i_x]
        ubd[i_x - 1] = ubd[i_x - 1] + __cse_0d
        ub[i_x - 1] = ub[i_x - 1] + __cse_0
        ubd[i_x] = ubd[i_x] + 2.0 * -__cse_0d
        ub[i_x] = ub[i_x] + 2.0 * -__cse_0
        ubd[i_x + 1] = ubd[i_x + 1] + __cse_0d
        ub[i_x + 1] = ub[i_x + 1] + __cse_0
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    return nothing
end

function stencil_loss(loss, u, w, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_x2 = 2:i_n - 1
        loss[1] = loss[1] + w[i_x2] ^ 2
    end
end
