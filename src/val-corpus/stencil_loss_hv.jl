function initstacks_stencil_loss_b()
    w_stack = Vector{Float64}()
    return w_stack
end

function stencil_loss_hv(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd, w_stack)
    w_stack_d = Vector{Float64}()
    for i_x = 2:i_n - 1
        push!(w_stack_d, wd[i_x])
        push!(w_stack, w[i_x])
        wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n - 1:-1:2
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    for i_x = i_n - 1:-1:2
        wd[i_x] = pop!(w_stack_d)
        w[i_x] = pop!(w_stack)
        ubd[i_x - 1] = ubd[i_x - 1] + wbd[i_x]
        ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
        ubd[i_x] = ubd[i_x] + 2.0 * -(wbd[i_x])
        ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
        ubd[i_x + 1] = ubd[i_x + 1] + wbd[i_x]
        ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    return nothing
end

function stencil_loss(loss, u, w, i_n)
    for i_x = 2:i_n - 1
        w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    end
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
end
