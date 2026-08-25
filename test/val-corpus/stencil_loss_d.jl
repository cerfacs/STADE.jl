function stencil_loss_d(loss, lossd, u, ud, w, wd, i_n)
    for i_x = 2:i_n - 1
        wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_x2 = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_x2]) * wd[i_x2]
        loss[1] = loss[1] + w[i_x2] ^ 2
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
