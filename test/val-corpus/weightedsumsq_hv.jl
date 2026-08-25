function initstacks_weightedsumsq_b()
    return nothing
end

function weightedsumsq_hv(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    for i_x = 1:i_n
        lossd[1] = lossd[1] + (u[i_x] ^ 2 * wd[i_x] + w[i_x] * ((2 * u[i_x]) * ud[i_x]))
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        wbd[i_x] = wbd[i_x] + (lossb[1] * ((2 * u[i_x]) * ud[i_x]) + u[i_x] ^ 2 * lossbd[1])
        wb[i_x] = wb[i_x] + u[i_x] ^ 2 * lossb[1]
        ubd[i_x] = ubd[i_x] + ((w[i_x] * lossb[1]) * (2 * ud[i_x]) + (2 * u[i_x]) * (lossb[1] * wd[i_x] + w[i_x] * lossbd[1]))
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * (w[i_x] * lossb[1])
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
end
