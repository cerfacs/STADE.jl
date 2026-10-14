function initstacks_red_single_cube_b()
    return nothing
end

function red_single_cube_hv(loss, lossb, u, ub, i_n, lossd, lossbd, ud, ubd)
    for i_x = 1:i_n
        __hcse_0 = u[i_x]
        lossd[1] = lossd[1] + (3 * __hcse_0 ^ 2) * ud[i_x]
        loss[1] = loss[1] + __hcse_0 ^ 3
    end
    for i_x = i_n:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = u[i_x]
        __hcse_3 = 3 * __hcse_2 ^ 2
        ubd[i_x] = ubd[i_x] + (__hcse_1 * (3 * ((2__hcse_2) * ud[i_x])) + __hcse_3 * lossbd[1])
        ub[i_x] = ub[i_x] + __hcse_3 * __hcse_1
    end
    return nothing
end

function red_single_cube(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 3
    end
end
