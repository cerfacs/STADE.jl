function initstacks_stencil_loss_b()
    w_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (w_stack, loss_stack)
end

function stencil_loss_b(loss, lossb, u, ub, w, wb, i_n, w_stack, loss_stack)
    for i_x = 2:i_n - 1
        push!(w_stack, w[i_x])
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n - 1:-1:2
        loss[1] = pop!(loss_stack)
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    for i_x = i_n - 1:-1:2
        w[i_x] = pop!(w_stack)
        ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
        ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
        ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
        wb[i_x] = 0.0
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
