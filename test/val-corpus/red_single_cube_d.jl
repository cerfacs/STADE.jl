function red_single_cube_d(loss, lossd, u, ud, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        lossd[1] = lossd[1] + (3 * __cse_0 ^ 2) * ud[i_x]
        loss[1] = loss[1] + __cse_0 ^ 3
    end
    return nothing
end

function red_single_cube(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 3
    end
end
