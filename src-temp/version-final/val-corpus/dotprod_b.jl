function initstacks_dotprod_b()
    loss_stack = Vector{Float64}()
    return loss_stack
end

function dotprod_b(loss, lossb, u, ub, v, vb, i_n, loss_stack)
    for i_seq_x = 1:i_n
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        loss[1] = pop!(loss_stack)
        ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
        vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    end
    return nothing
end

function dotprod(loss, u, v, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 1:i_n
        #= none:3 =#
        loss[1] = loss[1] + u[i_seq_x] * v[i_seq_x]
        #= none:4 =#
    end
end
