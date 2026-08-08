function initstacks_stencil_loss_b()
    return
end
function stencil_loss_b(loss, lossb, u, ub, w, wb, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = i_n - 1:-1:2
        wb[i_seq_x] = wb[i_seq_x] + 2 * w[i_seq_x] * lossb[1]
    end
    for i_x = 2:i_n - 1
        ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
        ub[i_x] = ub[i_x] - 2.0 * wb[i_x]
        ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
        wb[i_x] = 0.0
    end
    return 
end
function stencil_loss(loss, u, w, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end
