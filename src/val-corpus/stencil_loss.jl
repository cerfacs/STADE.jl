function stencil_loss(loss, u, w, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end
