function initstacks_pipeline_b()
    loss_stack = Vector{Float64}()
    return loss_stack
end

function pipeline_b(loss, lossb, u, ub, v, vb, w, wb, i_n, loss_stack)
    for i_x = 1:i_n
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_seq_x = 1:i_n
        push!(loss_stack, loss[1])
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        loss[1] = pop!(loss_stack)
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
        ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
        wb[i_x] = 0.0
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vb[i_x] = 0.0
    end
    return nothing
end

function pipeline(loss, u, v, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        v[i_x] = u[i_x] ^ 2 + 1.0
        #= none:4 =#
    end
    #= none:5 =#
    for i_x = 1:i_n
        #= none:6 =#
        w[i_x] = v[i_x] * u[i_x]
        #= none:7 =#
    end
    #= none:8 =#
    for i_seq_x = 1:i_n
        #= none:9 =#
        loss[1] = loss[1] + w[i_seq_x]
        #= none:10 =#
    end
end
