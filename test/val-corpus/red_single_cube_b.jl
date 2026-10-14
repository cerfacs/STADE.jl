function initstacks_red_single_cube_b()
    return nothing
end

function red_single_cube_b(loss, lossb, u, ub, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 3
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + (3 * u[i_x] ^ 2) * lossb[1]
    end
    return nothing
end

function red_single_cube(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 3
    end
end
