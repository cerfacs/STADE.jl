function stencil_loss_d(loss, lossd, u, ud, w, wd, i_n)
    for i_x = 2:i_n - 1
        wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end

function stencil_loss(loss, u, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 2:i_n - 1
        #= none:3 =#
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 2:i_n - 1
        #= none:6 =#
        loss[1] = loss[1] + w[i_seq_x] ^ 2
        #= none:7 =#
    end
end
