function red_refuse_invariant_d(loss, lossd, u, ud, c, cd, i_n)
    for i_x = 1:i_n
        __cse_0 = c[1]
        __cse_1 = u[i_x]
        lossd[1] = lossd[1] + (__cse_0 * ud[i_x] + __cse_1 * cd[1])
        loss[1] = loss[1] + __cse_1 * __cse_0
    end
    return nothing
end

function red_refuse_invariant(loss, u, c, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * c[1]
    end
end
