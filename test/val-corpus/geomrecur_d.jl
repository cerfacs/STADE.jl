function geomrecur_d(loss, lossd, u, ud, c, cd, i_n)
    for i_x = 2:i_n
        __cse_0 = u[i_x - 1]
        ud[i_x] = __cse_0 * cd + c * ud[i_x - 1]
        u[i_x] = c * __cse_0
    end
    for i_x = 1:i_n
        __cse_1 = u[i_x]
        lossd[1] = lossd[1] + (2__cse_1) * ud[i_x]
        loss[1] = loss[1] + __cse_1 ^ 2
    end
    return nothing
end

function geomrecur(loss, u, c, i_n)
    for i_x = 2:i_n
        u[i_x] = c * u[i_x - 1]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
end
