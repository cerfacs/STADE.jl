function initstacks_normcomp_b()
    w_stack = Vector{Float64}()
    loss_stack = Vector{Float64}()
    return (w_stack, loss_stack)
end

function normcomp_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd, w_stack, loss_stack)
    w_stack_d = Vector{Float64}()
    loss_stack_d = Vector{Float64}()
    for i_x = 1:i_n
        push!(w_stack_d, wd[i_x])
        push!(w_stack, w[i_x])
        wd[i_x] = ud[i_x] + -(vd[i_x])
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        push!(loss_stack_d, lossd[1])
        push!(loss_stack, loss[1])
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        lossd[1] = pop!(loss_stack_d)
        loss[1] = pop!(loss_stack)
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    for i_x = i_n:-1:1
        wd[i_x] = pop!(w_stack_d)
        w[i_x] = pop!(w_stack)
        ubd[i_x] = ubd[i_x] + wbd[i_x]
        ub[i_x] = ub[i_x] + wb[i_x]
        vbd[i_x] = vbd[i_x] + -(wbd[i_x])
        vb[i_x] = vb[i_x] + -(wb[i_x])
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    return nothing
end

function normcomp(loss, u, v, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        w[i_x] = u[i_x] - v[i_x]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 1:i_n
        #= none:6 =#
        loss[1] = loss[1] + w[i_seq_x] ^ 2
        #= none:7 =#
    end
end
