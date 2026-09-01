function normcomp_d(loss, lossd, u, ud, v, vd, w, wd, i_n)
    for i_x = 1:i_n
        wd[i_x] = ud[i_x] + -(vd[i_x])
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_x2 = 1:i_n
        __cse_0 = w[i_x2]
        lossd[1] = lossd[1] + (2__cse_0) * wd[i_x2]
        loss[1] = loss[1] + __cse_0 ^ 2
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
