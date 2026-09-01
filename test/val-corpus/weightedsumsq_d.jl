function weightedsumsq_d(loss, lossd, u, ud, w, wd, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        __cse_1 = __cse_0 ^ 2
        __cse_2 = w[i_x]
        lossd[1] = lossd[1] + (__cse_1 * wd[i_x] + __cse_2 * ((2__cse_0) * ud[i_x]))
        loss[1] = loss[1] + __cse_2 * __cse_1
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
end
