function initstacks_stencil_loss_b()
    return nothing
end

function stencil_loss_b(loss, lossb, u, ub, w, wb, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_x2 = 2:i_n - 1
        loss[1] = loss[1] + w[i_x2] ^ 2
    end
    for i_x2 = i_n - 1:-1:2
        wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    end
    for i_x = i_n - 1:-1:2
        __oldb_0 = wb[i_x]
        wb[i_x] = 0.0
        ub[i_x - 1] = ub[i_x - 1] + __oldb_0
        ub[i_x] = ub[i_x] + 2.0 * -__oldb_0
        ub[i_x + 1] = ub[i_x + 1] + __oldb_0
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
